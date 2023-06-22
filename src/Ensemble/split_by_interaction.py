import pandas as pd
from collections import defaultdict
import os
import argparse
import csv
from datetime import datetime
import pytz


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


def ensemble_interaction(
    train_df: pd.DataFrame,
    less_df: pd.DataFrame,
    much_df: pd.DataFrame,
    num_interaction: int,
) -> None:
    """
    Perform ensemble interaction based on the number of interactions for each user,
    concatenate the resulting DataFrames, and save the output to a CSV file.

    Args:
        train_df (pd.DataFrame): DataFrame containing training data.
        less_df (pd.DataFrame): DataFrame containing results of models that behave well for data with less interaction.
        much_df (pd.DataFrame): DataFrame containing results of models tthat behave well for data with a lot of interaction.
        num_interaction (int): Threshold for the number of interactions to split the users.

    Returns:
        None
    """

    grouped = train_df.groupby("user").size().reset_index(name="num_inter")

    split_under = grouped[grouped["num_inter"] <= num_interaction][
        "user"
    ].values.tolist()

    split_over = grouped[grouped["num_inter"] > num_interaction]["user"].values.tolist()

    less_df = less_df[less_df["user"].isin(split_under)]

    much_df = much_df[much_df["user"].isin(split_over)]

    output = pd.concat([less_df, much_df])
    output.sort_index()

    # Create the output folder
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    date_time = korea_date_time()
    file_name = f"{output_folder}/ensemble-{date_time}.csv"
    output.to_csv(file_name, index=False)

    print(f"{file_name} is successfully saved!")


def main(args):
    train_df = pd.read_csv(args.train_path)
    less_df = pd.read_csv(f"{args.file_path}{args.less_file}")
    much_df = pd.read_csv(f"{args.file_path}{args.much_file}")

    ensemble_interaction(
        train_df=train_df,
        less_df=less_df,
        much_df=much_df,
        num_interaction=args.num_interaction,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_path", type=str, default="../../data/train/train_ratings.csv"
    )
    parser.add_argument("--file_path", type=str, default="for_ensemble/")
    parser.add_argument("--less_file", type=str, default="output1.csv")
    parser.add_argument("--much_file", type=str, default="output2.csv")
    # 유저를 절반: 114, 상호작용 횟수 분포 25%: 265,50%: 499,75%: 749
    parser.add_argument("--num_interaction", type=int, default=114)

    args = parser.parse_args()
    main(args)
