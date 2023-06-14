from pathlib import Path
import pandas as pd
from typing import Tuple
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
import sys


def data_preprocessing(ratings_data: pd.DataFrame) -> pd.DataFrame:
    ratings_data = ratings_data.copy()

    ratings_data["user"] = ratings_data["user"].astype("category")
    ratings_data["item"] = ratings_data["item"].astype("category")
    ratings_data["user_id"] = ratings_data["user"].cat.codes
    ratings_data["item_id"] = ratings_data["item"].cat.codes

    return ratings_data


def create_user_item_map(ratings_data: pd.DataFrame) -> Tuple[dict, dict, dict, dict]:
    user_id_to_user_map = dict(
        enumerate(ratings_data["user"].cat.categories)
    )  # 새로운 user_id => 기존 CustomerID
    item_id_to_item_map = dict(
        enumerate(ratings_data["item"].cat.categories)
    )  # 새로운 item_id => 기존 StockCode
    user_to_user_id_map = dict()  # 기존 user => 새로운 user_id
    item_to_item_id_map = dict()  # 기존 item  => 새로운 item_id

    for x, y in zip(user_id_to_user_map.keys(), user_id_to_user_map.values()):
        user_to_user_id_map[y] = x

    for x, y in zip(item_id_to_item_map.keys(), item_id_to_item_map.values()):
        item_to_item_id_map[y] = x

    return (
        user_id_to_user_map,
        item_id_to_item_map,
        user_to_user_id_map,
        item_to_item_id_map,
    )


def create_sparse_matrix(ratings_data: pd.DataFrame) -> csr_matrix:
    ratings_data = ratings_data.copy()
    df = ratings_data.pivot_table(
        ["time"],
        index=ratings_data["user_id"],
        columns=ratings_data["item_id"],
        aggfunc="count",
        fill_value=0,
    )
    sparse_user_item = sparse.csr_matrix(df)

    assert ratings_data["user_id"].nunique() == sparse_user_item.shape[0]
    assert ratings_data["item_id"].nunique() == sparse_user_item.shape[1]

    return sparse_user_item


def load_data(isTrain: bool = True) -> Tuple[pd.DataFrame, csr_matrix]:
    file_path = Path.cwd() / "data"
    train_path = file_path / "train"
    eval_path = file_path / "eval"

    ratings_data = pd.read_csv(train_path / "train_ratings.csv")

    print("prepare data preprocessing")
    ratings_data = data_preprocessing(ratings_data)

    print("create csr matrix")
    sparse_matrix = create_sparse_matrix(ratings_data)

    return ratings_data, sparse_matrix
