import numpy as np
import scipy.sparse as sp

class EASE:     
    def __init__(self, _lambda):
        self.B = None
        self._lambda = _lambda
        
    def train(self, X):
        """
        Trains the EASE model.

        Parameters:
        - X (scipy.sparse.csr_matrix): Interaction matrix of shape (user_num, item_num).

        """

        # Compute the G matrix
        G = X.T @ X  # item_num * item_num
        G += self._lambda * sp.identity(G.shape[0])  # Regularization term
        G = G.todense()  # Convert to a dense matrix

        # Compute the P matrix
        P = np.linalg.inv(G)

        # Compute the B matrix (item similarity matrix)
        self.B = -P / np.diag(P)  # equation 8 in the paper: B_{ij}=0 if i = j else -\frac{P_{ij}}{P_{jj}}
        np.fill_diagonal(self.B, 0.)  # Set diagonal elements to zero

        # Store the item similarity matrix and interaction matrix
        self.item_similarity = np.array(self.B)  # item_num * item_num
        self.interaction_matrix = X  # user_num * item_num
    
    def forward(self, user_row):
        """
        Compute the recommendation scores for a given user.

        Parameters:
        - user_row (numpy.ndarray): Row vector representing the user's interactions with items.

        Returns:
        - numpy.ndarray: Recommendation scores for each item.

        """
        return user_row @ self.B
