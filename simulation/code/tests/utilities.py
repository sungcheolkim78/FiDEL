
import os
import numpy as np
import matplotlib.pyplot as plt

    
class BaseContainer:
    """Container used for storing values.

    Allocates space using np.zeros for the number of values expected to store,
    while maintaining the number of values inputed by the _next attribute.

    Use this class by subclassing and defining and append method.
    """
    def __init__(self):
        self._vals = np.zeros(self._size)
        self._next = 0

    @property
    def vals(self):
        """Return ndarray of filled values."""
        return self._vals[:self._next]

    def append(self, x):
        raise NotImplementedError


class VectorContainer(BaseContainer):
    """Container for concatinating many vectors

    Args:
        Nreps: (int) number of repeated vectors to store
        Nvec: (int) size of vector to repeated store
    """
    def __init__(self, Nreps, Nvec):
        self._size = Nreps * Nvec
        super().__init__()

    def append(self, x):
        """Concatenate all preceding vectors with vector x.

        Args:
            x: ((Nvec,) ndarray)
        """
        if x.ndim !=1:
            raise ValueError("Require that x is a 1-d ndarray")

        M = x.size

        if self._next + M > self._size:
            raise ValueError("Input data too big.")

        for val in x:
            self._vals[self._next] = val
            self._next += 1


class MatrixContainer(BaseContainer):
    """Container for concatinating upper diagonal elements of a matrix.

    Args:
        Nreps: (int) number of repeated vectors to store
        Nmat: (int) number of rows, and consequently columns of matrix,
            it is assumed that the matrix is square.
    """
    def __init__(self, Nreps, Nmat):
        self._size = int(Nreps * Nmat * (Nmat + 1) / 2)
        super().__init__()

    def append(self, x):
        """Concatenate all preceding matrix elements with those of matrix x.

        Args:
            x: ((Nmat, Nmat) ndarray)
        """
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError("Input must be a square matrix")

        M = x.shape[0]
        if self._next + M*(M+1)/2 > self._size:
            raise ValueError("input data too big.")

        for i in range(M):
            for j in range(i, M):
                self._vals[self._next] = x[i, j]
                self._next += 1


def hist(x, targets, target_labels, xlabel, savename, bins=None):
    """Plot histogram and target values.

    Histogram is saved as a pdf to file specified by savename.

    Args:
        x: ((N,) ndarray) of values to compute histogram
        targets: (length n container of numbers) representing the target values
            these values will be designated as vertical lines.
        target_labels: (length n container of str) used for figure legend
            to label the target lines with the corresponding target values. 
        xlabel: (str) x-axis label
        savename: (str) path and filename in which histogram plot is printed.
        bins: any viable input for the bins argument of matplotlib.pyplot.hist
    """
    n_targets = len(targets)
    if n_targets != len(target_labels):
        raise ValueError("Number of targets must equal number of target labels")

    cmap = plt.get_cmap("viridis")

    plt.figure(figsize=(3.5, 3))

    plt.hist(x, bins=bins)
    ylim = plt.ylim()

    for i, t in enumerate(targets):
        plt.plot([t, t], ylim, ":", 
                color=cmap(i/(n_targets-1)), 
                label=target_labels[i])

    plt.legend(loc=0)
    plt.ylim(ylim)

    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(savename, fmt=savename.split(".")[-1])


    

