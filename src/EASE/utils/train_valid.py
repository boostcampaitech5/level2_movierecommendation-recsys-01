import os
import numpy as np
import pandas as pd

def train_valid_split(train_df, num_seq, num_ran, random_seed=42):
    """
    Split the train dataframe into train and validation sets.

    Args:
        train_df (pandas.DataFrame): DataFrame containing the training data.
        num_seq (int): Number of data points per user to be included in the validation set.
        num_ran (int): Number of randomly selected previous data points per user to be included in the validation set.
        random_seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing train data and validation data as pandas DataFrames.
               The train data DataFrame contains the remaining data points after excluding the validation data.
               The validation data DataFrame consists of the last num_seq data points and num_ran randomly selected previous data points per user.
               Both DataFrames are sorted by the 'user' column.

    Raises:
        AssertionError: If the sum of num_seq and num_ran is not equal to 10.

    """
    # Set random seed
    np.random.seed(random_seed)

    total_num = 10
    assert np.isclose(num_seq + num_ran, total_num), "The sum of num_seq and num_ran should be equal to 10."


    # Extract the last num_seq data points per user to create valid data
    valid_data_last = train_df.groupby('user').tail(num_seq).copy()

    # Exclude valid data from train data
    train_data = train_df[~train_df.index.isin(valid_data_last.index)].copy()

    # Randomly select num_ran previous data points per user to create random_data
    random_data = train_data.groupby('user').apply(lambda x: x.sample(num_ran)).reset_index(drop=True)

    # Exclude random_data from train_data based on matching user and item values
    train_data = train_data[~train_data[['user', 'item']].apply(tuple, axis=1).isin(random_data[['user', 'item']].apply(tuple, axis=1))].copy()

    valid_data = pd.concat([valid_data_last, random_data], ignore_index=True)
    valid_data = valid_data[['user', 'item']].sort_values('user')

    return train_data, valid_data
