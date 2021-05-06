"""
This module was taken from the moca package licensed under the 
Apache 2.0 License

Tools for determining whether data follows MOCA convention.

Available Functions:
- validate_rank_data
- validate_label_data
"""
# TODO figure out appropriate referencing of code from a different package
import numpy as np

def _is_disjoint(x, y):
    """Are the ndarray x and y disjoint?"""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1-d ndarrays")

    test_val = np.setdiff1d(x, y).size
    test_val += np.setdiff1d(y, x).size

    return test_val == 0

def validate_rank_data(R):
    """Test whether data are ndarray of sample rank.

    Args:
        R : ((N,) or (M, N) ndarray) sample rank predictions, no ties.  For 
            2-d data, sample ranks are independently assigned for each row.
    Raises:
        TypeError: if data are not np.ndarray
        ValueError: for empty arrays
        ValueError: if data are not a 1 or 2d np.ndarray
        ValueError: if data are not rank
    """
    if not isinstance(R, np.ndarray):
        raise TypeError("Data must be an ndarray.")

    if R.size == 0:
        raise ValueError("Input data is empty")

    if R.ndim == 1:

        true_r = np.arange(1, R.size+1)

        if not _is_disjoint(true_r, R):
            raise ValueError(("Base classifier predictions must be an ndarray of "
                            "sample rank values."))
    elif R.ndim == 2:

        true_r = np.arange(1, R.shape[1] + 1)

        for i in range(R.shape[0]):
            if not _is_disjoint(true_r, R[i, :]):
                raise ValueError(("Base classifier predictions must "
                            "be sample ranks."))
    else:
        raise ValueError("Data must be a 1-d or 2-d ndarray.")

def validate_label_data(y):
    """Test if labels satisfy convention.

    Args:
        y : (N,) ndarray of values

    Raises:
        TypeError: if y is not ndarray
        ValueError: if not binary (0, 1) values
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("Input must be ndarray.")

    if y.ndim != 1:
        raise ValueError("Labels be 1-dimensional, i.e. have shape (N,)")

    true_y = np.array([0,1])

    if not _is_disjoint(true_y, y):
        raise ValueError("Label data must be binary (0,1) values.")


