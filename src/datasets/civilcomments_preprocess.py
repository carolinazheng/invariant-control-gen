import argparse
import os
import pickle
import spacy
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def filter_no_identity(df):
    return df[df.identity_annotator_count > 0].reset_index(drop=True)


def set_environments(df, colnames, n_envs=2):
    for colname in colnames:
        if n_envs == 2:
            df[f"env_{colname}"] = (df[colname] > df[colname].median()).astype(int)
        elif n_envs == 3:
            q1 = df[colname].quantile(0.33)
            if colname == "created_date":
                q2 = df[colname].quantile(0.66) + pd.Timedelta(seconds=1)
                bins = [pd.Timestamp.min, q1, q2, pd.Timestamp.max]
            else:
                q2 = df[colname].quantile(0.66) + 1e-7
                bins = [-float("inf"), q1, q2, float("inf")]
            df[f"env_{colname}"] = pd.cut(
                df[colname], bins=bins, labels=[0, 1, 2], include_lowest=True
            )
        else:
            raise Exception("Only supporting 2 or 3 environments.")

    return df


def make_Xy_bows(df, nlp, cv=None):
    def preprocess_text(text):
        doc = nlp(text.lower())
        tokens = [
            tok.lemma_
            for tok in doc
            if not tok.is_punct and not tok.is_stop and tok.text.isalpha()
        ]
        return " ".join(tokens)
    start = time.time()

    tokenized = df["comment_text"].transform(preprocess_text)

    if cv is None:
        cv = CountVectorizer(max_features=1000)
        X = cv.fit_transform(tokenized).toarray()
    else:
        X = cv.transform(tokenized).toarray()

    y = (df["target"] > 0.5).astype(int).to_numpy()
    print(f"Took {round(time.time() - start, 2)}s to process {len(df)} examples.")
    return X, y, cv


def make_Xy_metadata(df, covariates, scaler=None):
    start = time.time()

    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(df[covariates].values)

    X = scaler.transform(df[covariates].values)
    y = (df["target"] > 0.5).astype(int).to_numpy()
    print(f"Took {round(time.time() - start, 2)}s to process {len(df)} examples.")

    return X, y, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--n_envs", type=int, default=2)
    parser.add_argument("--n_total", type=int, default=28000)
    parser.add_argument("--n_test", type=int, default=4200)
    parser.add_argument("--balanced", type=str_to_bool, default="0")
    parser.add_argument("--load_evian", type=str_to_bool, default="0")
    parser.add_argument("--evian_model_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")  # python -m spacy download en_core_web_sm
    np.random.seed(args.seed)

    df = pd.read_csv(f"{args.data_dir}/df_train_all.csv")
    test_pub_df = pd.read_csv(f"{args.data_dir}/test_public_expanded.csv")
    test_priv_df = pd.read_csv(f"{args.data_dir}/test_private_expanded.csv")

    clean_df = filter_no_identity(df)
    clean_test_pub_df = filter_no_identity(test_pub_df).rename(
        columns={"toxicity": "target"}
    )
    clean_test_priv_df = filter_no_identity(test_priv_df).rename(
        columns={"toxicity": "target"}
    )

    # This df contains all examples where we have identity annotations
    all_clean_df = pd.concat(
        [clean_df, clean_test_pub_df, clean_test_priv_df]
    ).reset_index(drop=True)

    # Subsample from CivilComments
    if args.balanced:
        all_clean_df_toxic = all_clean_df[all_clean_df["target"] > 0.5].sample(
            args.n_total // 2, random_state=args.seed
        )
        all_clean_df_nontoxic = all_clean_df[all_clean_df["target"] <= 0.5].sample(
            args.n_total // 2, random_state=args.seed
        )
        all_clean_df_sub = pd.concat(
            [all_clean_df_toxic, all_clean_df_nontoxic]
        ).reset_index(drop=True)
    else:
        all_clean_df_sub = all_clean_df.sample(
            args.n_total, random_state=args.seed
        ).reset_index(drop=True)

    print(
        f"% toxic: {len(all_clean_df_sub[all_clean_df_sub['target'] > 0.5]) / len(all_clean_df_sub)}"
    )

    # Create train+valid and test split
    train_val_df, test_df = train_test_split(
        all_clean_df_sub, test_size=args.n_test, random_state=args.seed
    )

    # Create the auxiliary data environments
    attribute_cols = [
        "asian",
        "atheist",
        "bisexual",
        "black",
        "buddhist",
        "christian",
        "female",
        "heterosexual",
        "hindu",
        "homosexual_gay_or_lesbian",
        "intellectual_or_learning_disability",
        "jewish",
        "latino",
        "male",
        "muslim",
        "other_disability",
        "other_gender",
        "other_race_or_ethnicity",
        "other_religion",
        "other_sexual_orientation",
        "physical_disability",
        "psychiatric_or_mental_illness",
        "transgender",
        "white",
    ]

    train_val_df["attribute_sum"] = train_val_df[attribute_cols].sum(axis=1)
    train_val_df["created_date"] = pd.to_datetime(
        train_val_df["created_date"], utc=True
    )
    train_val_df["created_date"] = train_val_df["created_date"].dt.tz_localize(None)
    train_val_df = set_environments(
        train_val_df,
        ["identity_annotator_count", "created_date", "attribute_sum"],
        n_envs=args.n_envs,
    )

    # Train EviaN-Scramble model and create environments
    name_str = "balanced" if args.balanced else "unbalanced"

    if args.load_evian:
        with open(
            os.path.join(args.evian_model_dir, f"scramble_{name_str}.pkl"), "rb"
        ) as f:
            saved = pickle.load(f)

        clf = saved["clf"]
        X_train, y_train = saved["X_train"], saved["y_train"]
        X_test, y_test = saved["X_test"], saved["y_test"]

    else:
        X_train, y_train, cv = make_Xy_bows(train_val_df, nlp)
        X_test, y_test, _ = make_Xy_bows(test_df, nlp, cv=cv)

        clf = LogisticRegression(random_state=args.seed).fit(X_train, y_train)

        saved = {
            "clf": clf,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        if args.evian_model_dir is not None:
            with open(
                os.path.join(args.evian_model_dir, f"scramble_{name_str}.pkl"), "wb"
            ) as f:
                pickle.dump(saved, f)

    train_val_df["evian_bows"] = clf.predict_proba(X_train)[:, 1]
    train_val_df = set_environments(train_val_df, ["evian_bows"], n_envs=args.n_envs)

    # Train EviaN-Metadata model and create environments
    covariates = [
        "asian",
        "atheist",
        "bisexual",
        "black",
        "buddhist",
        "christian",
        "female",
        "heterosexual",
        "hindu",
        "homosexual_gay_or_lesbian",
        "intellectual_or_learning_disability",
        "jewish",
        "latino",
        "male",
        "muslim",
        "other_disability",
        "other_gender",
        "other_race_or_ethnicity",
        "other_religion",
        "other_sexual_orientation",
        "physical_disability",
        "psychiatric_or_mental_illness",
        "transgender",
        "white",
        "sexual_explicit",
    ]

    train_val_df["rating_binary"] = train_val_df["rating"].transform(
        lambda t: t == "approved"
    )
    test_df["rating_binary"] = test_df["rating"].transform(lambda t: t == "approved")

    if args.load_evian:
        with open(
            os.path.join(args.evian_model_dir, f"metadata_{name_str}.pkl"), "rb"
        ) as f:
            saved = pickle.load(f)

        clf = saved["clf"]
        X_train, y_train = saved["X_train"], saved["y_train"]
        X_test, y_test = saved["X_test"], saved["y_test"]
    else:
        X_train, y_train, scaler = make_Xy_metadata(train_val_df, covariates)
        X_test, y_test, _ = make_Xy_metadata(test_df, covariates, scaler=scaler)

        clf = LogisticRegression(random_state=args.seed).fit(X_train, y_train)

        saved = {
            "clf": clf,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        if args.evian_model_dir is not None:
            with open(
                os.path.join(args.evian_model_dir, f"metadata_{name_str}.pkl"), "wb"
            ) as f:
                pickle.dump(saved, f)

    train_val_df["evian_metadata"] = clf.predict_proba(X_train)[:, 1]
    train_val_df = set_environments(
        train_val_df, ["evian_metadata"], n_envs=args.n_envs
    )

    # Randomly select examples to reassign EviaN-Metadata environment to ensure that there are examples from each environment in the training set.
    if (
        args.n_envs == 3
        and len(train_val_df[train_val_df["env_evian_metadata"] == 1]) == 0
    ):
        print(train_val_df["env_evian_metadata"].value_counts())
        env_to_split = train_val_df["env_evian_metadata"].value_counts().argmax()
        df_to_split = train_val_df[train_val_df["env_evian_metadata"] == env_to_split]
        max_value = df_to_split["evian_metadata"].max()
        rows_to_split = df_to_split[df_to_split["evian_metadata"] == max_value].sample(
            len(df_to_split) // 2
        )
        print(f"Splitting env {env_to_split} with {len(df_to_split)} rows in half.")
        train_val_df.loc[rows_to_split.index, "env_evian_metadata"] = 1
        print(train_val_df["env_evian_metadata"].value_counts())

    # Create random split environment
    train_val_df["env_random"] = np.random.choice(
        range(args.n_envs), size=len(train_val_df)
    )

    # Create train-valid split
    if args.n_envs == 2:
        train_df, val_df = train_test_split(
            train_val_df, test_size=args.n_test, random_state=args.seed
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
    elif args.n_envs == 3:
        # Will be set in the training script based on env=1 for each environment type
        train_df = train_val_df
        val_df = None
        # Swap env 1 and 2 (training script will use the env label 2 as the validation environment for leave-one-environment-out validation)
        for colname in train_df.columns:
            if colname[:4] == "env_":
                train_df[colname] = train_df[colname].map(
                    lambda e: 2 if e == 1 else (1 if e == 2 else 0)
                )

    # Save preprocessed dataset
    for split, df in zip(["train", "valid", "test"], [train_df, val_df, test_df]):
        if df is None:
            print(f"Skipping {split}...")
        else:
            df.to_csv(os.path.join(args.save_dir, f"{split}.csv"), index=False)


if __name__ == "__main__":
    main()
