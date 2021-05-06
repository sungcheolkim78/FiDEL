"""Helper classes and functions for cl_performance_and_correlation.py.

Available functions:
- get_auc: returns an np.ndarray of auc values evenly spaced between
    specified AUC limits.
- extra_upper_diagonal: return the extra upper diagonal elements of the 
    square symmetric matrix as a ndarray
- compute_mean_cond_corr_matrix: compute the average class conditional
    correlation matrix from rank data and class labels

Available Classes:
- ReplicateData: helper class for storing replicate auc values, and computing
    the mean and sem
- DataForPlots: organize data for plotting
- StatsTable: helper class for organize mean and sem values given parameter
    indexes
"""

import numpy as np

def get_auc(m, auc_limits):
    """Gustavo's method for generating uniformly distributed auc values.

    Args:
        m: (int) > 2, representing the number of base classifiers
        auc_limits: (len(2) list, tuple, or ndarray, i.e. Sequence) 
            representing the (minimum AUC, maximum AUC) among the 
            set of base classifiers. Each AUC value must be on 
            interval (0, 1).

    Returns:
        ((m,) ndarray) of auc values.
    """
    if m < 2:
        raise ValueError(("The number of base classifiers, m, "
                        "must be greater than 2."))
    if len(auc_limits) != 2:
        raise ValueError("Must supply len(2) sequence of AUC values.")

    for auc_lim in auc_limits:
        if auc_lim < 0 or auc_lim > 1:
            raise ValueError("AUC values must be on interval [0,1].")

    auc_range = auc_limits[1] - auc_limits[0]

    auc = np.zeros(m)

    for i in range(m):
        auc[i] = auc_limits[0] + auc_range * i / (m-1)

    return auc

def extra_upper_diagonal(A):
    """Retrieve upper extra diagonal elements from a square symmetric matrix.

    Args:
        A: ((M, M) ndarray) representing a symmetric matrix

    Returns:
        out: ((M(M-1)/2,) ndarray) of values
    """
    if A.ndim != 2:
        raise ValueError("Input must be a square symmetric matrix.")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square symmetric matrix.")
    if (A != A.T).any():
        raise ValueError("Input must be a square symmetric matrix.")

    M = A.shape[0]

    out = np.zeros(int(M*(M-1)/2))
    l = 0
    for i in range(M):
        for j in range(i+1, M):
            out[l] = A[i, j]
            l += 1

    return out

def compute_mean_cond_corr_matrix(R, y):
    """Compute and return the mean conditional correlation matrix.

    Args:
        R: ((M, N) ndarray) independent rows of sample ranks, no ties in row 
        y: ((N,) ndarray) of sample class labels [0, 1].

    Returns:
        C: ((M, M) ndarray) average conditional correlation matrix
    """
    C = np.corrcoef(R[:, y == 0])
    C += np.corrcoef(R[:, y == 1])
    C = C / 2

    M = C.shape[0]

    for i in range(M):
        for j in range(i+1, M):
            C[j, i] = C[i, j]

    return C


class ReplicateData:
    """Store and compute statistics of experimental replicates.

    Args:
        nreps: (int) number of replicates
    """
    def __init__(self, nreps):
        if nreps < 2:
            raise ValueError("Number of replicates should be >= 2")
        self._size = nreps
        self._next = 0
        self._data = np.zeros(nreps)

    def __len__(self):
        return self._next

    @property
    def mean(self):
        """Mean over replicate values.

        Returns:
            (float) of the mean

        Raises:
            ZeroDivisionError if there are no replicate data entries
        """
        if self._next == 0:
            raise ZeroDivisionError("No data entries to compute mean.")
        elif self._next < self._size:
            print("Mean computed for less than total replicate number.")
            return np.mean(self._data[:self._next])

        return np.mean(self._data)

    @property
    def sem(self):
        """Standard error of mean over replicate values.

        The standard error of mean is computed as:

        sem = (unbiased standard deviation) / sqrt(N replicates)

        Returns:
            (float) of the standard error of mean

        Raises:
            ZeroDivisionError if there are either 0 or 1 replicate data
                entry.
        """
        if self._next < 2:
            raise ZeroDivisionError("Require >= 2 data points to compute S.E.M.")
        elif self._next < self._size:
            print("SEM computed for less than total replicate number.")
            return np.std(self._data[:self._next], ddof=1) / np.sqrt(len(self))
        
        return np.std(self._data[:self._next], ddof=1) / np.sqrt(len(self))

    def reset(self):
        self._data.fill(0)
        self._next = 0

    def append(self, val):
        """Add data value to replicates.

        Args:
            val: (int or float) to be added to replicate data set
        """
        if not isinstance(val, int) and not isinstance(val, float):
            raise ValueError("Input data must be an int or float.")

        self._data[self._next] = val

        self._next += 1


class DataForPlots:
    """Organize data for plotting.

    Args:
        label: (str) label to be used in figure legend
        color: (any matplotlib compatible color argument), 
            designates the line and marker color associated
            with these data.
    """
    def __init__(self, label, color):
        self.label = label
        self.color = color

        self._data_attributes = ("x", "xerr", "y", "yerr")

    def reset_data(self):
        """Restore all attributes specified in _data_attributes to None."""
        for attribute in self._data_attributes:
            setattr(self, attribute, None)

    def update_data(self, xerr=None, yerr=None, *,x,y):
        """Update attribute values specified in _data_attributes.

        Args:
            x: (N length sequence) x-axis values for plots,
            y: (N length sequence) y-axis values for plots,
            xerr: (N length sequence) optional, containing error along x-axis
            yerr: (N length sequence) optional, containing error along y-axis
        """
        self.reset_data()
        
        self.x = x
        self.y = y

        self.xerr=xerr
        self.yerr=yerr


class StatsTable:
    """Store and retrieve mean and sem of data.

    Properties:
    - mean: returns ndarray of means
    - sem: returns ndarray of standard error of means (s.e.m.)

    Available Methods:
    - update: add mean and sem over replicates to the position
           specified by given index
    """
    def __init__(self, row_size, col_size):
        self._means = np.zeros(shape=(row_size, col_size))
        self._sem = np.zeros(shape=(row_size, col_size))

    @property
    def mean(self):
        """Return means as ndarray"""
        return self._means

    @property
    def sem(self):
        """Return standard error of means as ndarray"""
        return self._sem

    def update(self, rep_obj, idx):
        """Add mean and sem to corresponding ndarray at idx.

        Args:
            rep_obj: Instance of ReplicateData
            idx: tuple of indices to store mean and sem values.
        """
        self._means[idx] = rep_obj.mean
        self._sem[idx] = rep_obj.sem



