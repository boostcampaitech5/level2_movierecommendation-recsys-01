import numpy as np
import pandas as pd
from typing import Tuple
from scipy.sparse import csr_matrix

def create_matrix_and_mappings(dataframe: pd.DataFrame) -> Tuple[csr_matrix, dict, dict]:
    """
    Create a CSR matrix and index mappings for users and items based on the given dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe containing user-item interactions.

    Returns:
        tuple[csr_matrix, dict, dict]: A tuple containing the CSR matrix representation of user-item interactions,
                                      the mapping of user indices to user values,
                                      and the mapping of item indices to item values.
    """

    # Extract unique users and items
    users = dataframe['user'].unique()
    items = dataframe['item'].unique()

    # Create user and item index mappings
    user_index = {user: i for i, user in enumerate(users)}
    item_index = {item: i for i, item in enumerate(items)}

    # Create CSR matrix
    row_indices = dataframe['user'].map(user_index)
    col_indices = dataframe['item'].map(item_index)
    values = np.ones(len(dataframe))
    matrix = csr_matrix((values, (row_indices, col_indices)), shape=(len(users), len(items)))

    # Create reverse mappings for user and item indices
    index_to_user = {i: user for user, i in user_index.items()}
    index_to_item = {i: item for item, i in item_index.items()}

    return matrix, index_to_user, index_to_item
