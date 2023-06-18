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


def ensemble_top_rating(
    main_df: pd.DataFrame, sub_df: pd.DataFrame, using_topk: int
) -> None:
    """
    Ensemble the top ratings from main_df and additional ratings from sub_df,
    based on the specified number (using_topk) of top ratings to consider.

    Args:
        main_df (pd.DataFrame): The main DataFrame containing ratings.
        sub_df (pd.DataFrame): The additional DataFrame containing ratings.
        using_topk (int): The number of top ratings to consider from main_df.

    Returns:
        None. Saves the ensemble result to a CSV file.

    Raises:
        AssertionError: If using_topk is not an integer or falls outside the range of 0 to 10 (inclusive).
    """
    assert (
        isinstance(using_topk, int) and 0 <= using_topk <= 10
    ), "using_topk should be an integer between 0 and 10 (inclusive)"

    # Select the top ratings from main_df
    main_df = main_df.groupby("user").head(using_topk)

    # Select additional ratings from sub_df that are not present in main_df
    sub_df = (
        pd.merge(main_df, sub_df, how="outer", indicator=True)
        .query('_merge == "right_only"')
        .drop(columns=["_merge"])
    )
    sub_df = sub_df.groupby("user").head(10 - using_topk)

    # Concatenate main_df and sub_df
    output = pd.concat([main_df, sub_df])
    output = output.sort_values("user")

    # Create the output folder
    output_folder = "./for_ensemble"
    os.makedirs(output_folder, exist_ok=True)

    # Save the ensemble result to a CSV file
    date_time = korea_date_time()
    file_name = f"{output_folder}/ensemble-{date_time}.csv"
    output.to_csv(file_name, index=False)

    print(f"{file_name} is successfully saved!")


def main(args):
    main_df = pd.read_csv(f"{args.file_path}{args.main_file}")
    sub_df = pd.read_csv(f"{args.file_path}{args.sub_file}")
    ensemble_top_rating(main_df=main_df, sub_df=sub_df, using_topk=args.using_topk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, default="for_ensemble/")
    parser.add_argument("--main_file", type=str, default="output1.csv")
    parser.add_argument("--sub_file", type=str, default="output2.csv")
    parser.add_argument("--using_topk", type=int, default=5)

    args = parser.parse_args()
    main(args)
