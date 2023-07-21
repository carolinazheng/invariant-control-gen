import argparse
import os
import openai
import numpy as np
import pandas as pd
import time
from tqdm import tqdm


def query_gpt(gpt_prompt, model="text-davinci-003"):
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=gpt_prompt,
            temperature=1,
            max_tokens=50,
            top_p=0.9,
            frequency_penalty=0.0,
        )
    except (
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.APIError,
        openai.error.Timeout,
    ) as err:
        print(f"Unexpected {err=}, {type(err)=}. sleeping for 1min")
        time.sleep(60)

    return response


def prompt_builder(prompt_context, profession, modifier):
    constraint_prompt1 = f"The statement is {modifier[1]} but still takes into account the profession of the speaker."
    main_prompt = f"In a heated conversation about {prompt_context}, the {profession} made the following creative, but {modifier[0]} statement. "
    gpt_prompt = main_prompt + constraint_prompt1

    return gpt_prompt


def save_df(save_dir, prompt_context, modifiers, profession, n_continuations):
    res = []

    for context in prompt_context:
        for m in modifiers:
            gpt_prompt = prompt_builder(context, profession, m)
            print(f"{context} + {m}")

            for _ in tqdm(range(n_continuations)):
                response = query_gpt(gpt_prompt)
                res.append(
                    {
                        "prompt": gpt_prompt,
                        "context": context,
                        "modifier": m,
                        "profession": profession,
                        "continuation": response["choices"][0]["text"].strip(),
                    }
                )
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(save_dir, f"{profession}.csv"), index=False)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--winogenerated_path", type=str)
    parser.add_argument("--n_continuations", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.api_key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    np.random.seed(args.seed)

    prompt_context = ["politics", "relationships", "sports", "religion"]
    modifiers = [
        ["controversial", "controversial"],
        ["hateful, offensive, and aggressive", "toxic"],
    ]

    df_jobs = pd.read_json(
        args.winogenerated_path,
        lines=True,
    )
    jobs = pd.unique(df_jobs["occupation"])
    sub_jobs = np.random.choice(jobs, size=args.n_jobs, replace=False)
    print(sub_jobs)

    dfs = []

    for p in range(args.n_jobs):
        profession = sub_jobs[p]
        print(f"[{p}] {profession}")

        if f"{profession}.csv" in os.listdir(args.save_dir):
            print("Found saved df.")
            df = pd.read_csv(os.path.join(args.save_dir, f"{profession}.csv"))
        else:
            df = save_df(
                args.save_dir,
                prompt_context,
                modifiers,
                profession,
                args.n_continuations,
            )

        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(args.save_dir, "all_professions.csv"), index=False)


if __name__ == "__main__":
    main()
