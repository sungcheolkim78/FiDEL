"""Statistics for binary classifiers

Available Functions:
- delta: compute the difference in class conditioned mean
- rank_2_auc: compute AUC from sample rank and class label data
"""

import numpy as np
from . import validate as val

def delta(r, y):
    """Compute empirical delta from rank and label data

    Delta := E[R | Y=0] - E[R | Y=1]

    Args:
        r: ((N,) ndarray) of sample rank predictions, no ties, on interval 
            [1, N], samples from positive class are assumed to have low rank
        y: ((N,) ndarray) of sample class labels [0, 1].

    Returns:
        (float) on the interval [-N/2, N/2] representing delta

    References:
        M.E. Ahsen, R.M. Vogel, and G.A. Stolovitzky. "Unsupervised 
        Evaluation of Weighted Aggregation of Ranked Classification."
        JMLR 20.166 (2019)
    """
    if r.shape != y.shape:
        raise ValueError("r and y must be ((N,) ndarray).")

    val.validate_rank_data(r)
    val.validate_label_data(y)

    return np.mean(r[y == 0]) - np.mean(r[y == 1])

def rank_2_auc(r, y):
    """Given rank predictions r and class labels y compute AUC

    Ahsen et al. showed that

    AUC = Delta / N + 1/2

    where Delta = E[R | Y=0] - E[R | Y=1], N is the number of samples.
    R is a random variable representing a base classifier's rank predictions,
    and Y is a random variable representing the two possible sample classes.

    Args:
        r: ((N,) ndarray) of sample rank predictions, no ties, on interval 
            [1, N], samples from positive class are assumed to have low rank
        y: ((N,) ndarray) of sample class labels [0, 1].

    Returns:
        (float) on the interval [0, 1] representing the AUC

    References:
        M.E. Ahsen, R.M. Vogel, and G.A. Stolovitzky. "Unsupervised 
        Evaluation of Weighted Aggregation of Ranked Classification."
        JMLR 20.166 (2019)
    """
    d = delta(r, y)
    N = r.size
    return d/N + 0.5


