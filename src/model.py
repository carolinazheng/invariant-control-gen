import torch
from transformers import BertModel
from torch.optim import AdamW
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_f1_score,
    binary_calibration_error,
)


class ToxicCommentTagger(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.lr_schedule = args.lr_schedule
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.pretrain = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.dropout = nn.Dropout(self.pretrain.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.pretrain.config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        if args.config not in {"vanilla", "v-rex", "mmd", "coral"}:
            raise Exception(f"Unknown config: {args.config}")

        self.config = args.config
        # note: only supports training with 2 environments
        self.n_envs = 1 if self.config == "vanilla" else 2
        self.beta = args.beta
        self.warmup_beta = args.warmup_beta
        self.save_hyperparameters()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrain(
            input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output

    def _shared_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        envs = batch["envs"]

        outputs, features = self(input_ids, attention_mask)
        losses = self.criterion(outputs, labels)
        loss_sums = torch.zeros(self.n_envs)
        env_counts = torch.zeros(self.n_envs)
        features_by_env = []

        for i in torch.unique(envs).int().tolist():
            loss_sums[i] = losses[envs == i].sum()
            env_counts[i] = (envs == i).sum()

        for i in range(self.n_envs):
            features_by_env.append(features[(envs == i).squeeze()])

        return loss_sums, env_counts, outputs, features_by_env

    def _shared_log_step(self, loss_terms, step):
        for loss_name, value in loss_terms.items():
            self.log(
                f"{step}/step/{loss_name}",
                value,
                logger=True,
                batch_size=self.batch_size,
                on_step=True,
                rank_zero_only=True,
            )

    # Code for MMD and CORAL is from DomainBed (https://arxiv.org/abs/2007.01434)
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def training_step(self, batch, batch_idx):
        loss_sums, env_counts, outputs, features_by_env = self._shared_step(
            batch, batch_idx
        )
        loss_terms = {}
        loss = 0
        env_in_batch = env_counts > 0

        cur_step = self.trainer.global_step
        total_steps = self.trainer.estimated_stepping_batches

        if cur_step < self.warmup_beta * total_steps:
            beta_scale = cur_step / (self.warmup_beta * total_steps)
        else:
            beta_scale = 1.0

        if self.config == "vanilla":
            loss = loss_sums[0] / env_counts[0]
        elif self.config == "v-rex":
            loss_means = loss_sums[env_in_batch] / env_counts[env_in_batch]
            loss_envs = loss_means.sum()
            loss_var = loss_means.var()

            if len(loss_means) < 2:
                loss_var = 0

            loss = loss_envs + beta_scale * self.beta * loss_var

            loss_terms["loss_envs"] = loss_envs
            loss_terms["loss_reg"] = loss_var
        elif self.config == "mmd" or self.config == "coral":
            loss_erm = loss_sums.sum() / env_counts.sum()

            if env_in_batch.sum() < 2:
                loss = loss_erm
            else:
                penalty_fn = self.mmd if self.config == "mmd" else self.coral
                penalty = penalty_fn(features_by_env[0], features_by_env[1])

                loss = loss_erm + beta_scale * self.beta * penalty

            loss_terms["loss_erm"] = loss_erm
            loss_terms["loss_reg"] = penalty

        loss_terms["loss"] = loss
        self._shared_log_step(loss_terms, "train")

        train_out = {
            "loss_sums": loss_sums.detach(),
            "env_counts": env_counts.detach(),
            "predictions": outputs.detach(),
            "labels": batch["labels"],
            "envs": batch["envs"],
        }
        self.training_step_outputs.append(train_out)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_sums, env_counts, outputs, _ = self._shared_step(batch, batch_idx)

        val_out = {
            "loss_sums": loss_sums,
            "env_counts": env_counts,
            "predictions": outputs,
            "labels": batch["labels"],
            "envs": batch["envs"],
        }
        self.validation_step_outputs.append(val_out)

        return val_out

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_sums, env_counts, outputs, _ = self._shared_step(batch, batch_idx)

        test_out = {
            "loss_sums": loss_sums,
            "env_counts": env_counts,
            "predictions": outputs,
            "labels": batch["labels"],
            "envs": batch["envs"],
        }
        self.test_step_outputs.append(test_out)

        return test_out

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            probs = self(input_ids, attention_mask)[0].sigmoid()

        return probs

    def _shared_epoch_end(self, outputs, step):
        loss_terms = {}
        loss_sums = sum([out["loss_sums"] for out in outputs])
        env_counts = sum([out["env_counts"] for out in outputs])

        loss_terms["loss"] = loss_sums.sum() / env_counts.sum()

        for name, value in loss_terms.items():
            self.log(
                f"{step}/{name}",
                value,
                logger=True,
                sync_dist=True,
                rank_zero_only=True,
            )

        y_logits = torch.cat([out["predictions"] for out in outputs]).squeeze()
        y_labels = torch.cat([out["labels"] for out in outputs])
        y_labels = (y_labels > 0.5).int().squeeze()

        metrics = {
            "acc": binary_accuracy(y_logits, y_labels),
            "f1_score": binary_f1_score(y_logits, y_labels),
            "auroc": binary_auroc(y_logits, y_labels),
            "ece": binary_calibration_error(y_logits, y_labels),
        }

        if self.config == "v-rex" and step != "test":
            envs = torch.cat([out["envs"] for out in outputs]).squeeze()
            env_idxs = torch.unique(envs).int().tolist()
            eces = [
                binary_calibration_error(y_logits[envs == i], y_labels[envs == i])
                for i in env_idxs
            ]
            metrics["ece_avg_env"] = sum(eces) / len(eces)

        for name, value in metrics.items():
            self.log(
                f"{step}/{name}",
                value,
                logger=True,
                sync_dist=True,
                rank_zero_only=True,
            )

    def on_train_epoch_end(self):
        self._shared_epoch_end(self.training_step_outputs, step="train")
        self.training_step_outputs.clear()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        self._shared_epoch_end(self.validation_step_outputs, step="val")
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        self._shared_epoch_end(self.test_step_outputs, step="test")
        self.test_step_outputs.clear()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return dict(
            optimizer=optimizer,
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
    ):
        if self.lr_schedule == "warmup_linear_decay":
            cur_step = self.trainer.global_step
            total_steps = self.trainer.estimated_stepping_batches

            if cur_step < 0.1 * total_steps:
                lr_scale = cur_step / (0.1 * total_steps)
            else:
                lr_scale = 1 - (cur_step - 0.1 * total_steps) / (0.9 * total_steps)

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        optimizer.step(closure=optimizer_closure)
