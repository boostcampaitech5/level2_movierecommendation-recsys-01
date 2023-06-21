import os
import ast
import csv
import sys
import pytz
from tqdm import tqdm
import pandas as pd
import argparse
from datetime import datetime
from collections import defaultdict


def korea_date_time():
    """
    Retrieves the current date and time in the Korea Standard Time (KST) timezone.

    Returns:
        str: The current date and time formatted as 'YYYY-MM-DD_HH:MM:SS' in KST.
    """
    korea_timezone = pytz.timezone("Asia/Seoul")
    date_time = datetime.now(tz=korea_timezone)
    date_time = date_time.strftime("%Y-%m-%d_%H:%M:%S")

    return date_time


def arg_as_list(value):
    # 이 함수는 문자열로 받은 값을 쉼표(,)를 기준으로 분할하여 리스트로 반환합니다.
    return [float(x) for x in value.split(",")]


def save_file(output: pd.DataFrame) -> None:
    # Create the output folder
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    # Save the ensemble result to a CSV file
    date_time = korea_date_time()
    file_name = f"{output_folder}/ensemble-{date_time}.csv"
    output.to_csv(file_name, index=False)

    print(f"{file_name} is successfully saved!")


def soft_voting(args):
    files = os.listdir(args.file_path)
    files.sort()

    df_list = [pd.read_csv(os.path.join(args.file_path, i)) for i in files]
    user_list = df_list[0]["user"].unique()
    df_len = len(df_list)
    ensemble_ratio = arg_as_list(args.weight)

    result = []
    tbar = tqdm(user_list, desc="Ensemble")

    for user in tbar:
        temp = defaultdict(float)
        for idx in range(df_len):
            items = df_list[idx][df_list[idx]["user"] == user]["item"].values

            for item_idx, item in enumerate(items):
                temp[item] += ensemble_ratio[idx] * (1 - item_idx / len(items))

        for key, _ in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:10]:
            result.append((user, key))

    output = pd.DataFrame(result, columns=["user", "item"])

    save_file(output)


def hard_voting(args) -> None:
    files = os.listdir(args.file_path)
    files.sort()

    df_list = [pd.read_csv(os.path.join(args.file_path, i)) for i in files]
    ensemble_num = [i * 10 for i in arg_as_list(args.weight)]
    for idx in range(len(df_list)):
        if idx == 0:
            df_list[idx] = df_list[idx].groupby("user").head(ensemble_num[idx])
        else:
            for iter in range(idx):
                df_list[idx] = (
                    pd.merge(df_list[iter], df_list[idx], how="outer", indicator=True)
                    .query('_merge == "right_only"')
                    .drop(columns=["_merge"])
                )
            df_list[idx] = df_list[idx].groupby("user").head(ensemble_num[idx])

    output = pd.concat(df_list)
    output = output.sort_values("user")

    save_file(output)


def main(args):
    if args.option.lower() == "soft":
        soft_voting(args)
    elif args.option.lower() == "hard":
        hard_voting(args)
    else:
        print("Wrong Option!!! Try Again")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--option", type=str, default="soft")
    parser.add_argument("--file_path", type=str, default="for_ensemble/")
    parser.add_argument("--weight", type=str, default="0.5,0.5")

    args = parser.parse_args()
    main(args)
