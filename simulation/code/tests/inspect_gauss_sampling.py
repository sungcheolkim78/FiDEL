"""Qualitative inspection of mean and covariance from multivariate Gaussian

Running this program will:

    1. Generate NREPS replicates of of N observations of an M dimensional random
        vector.
    2. For different covariance matrices, print two plots:
        i. Histogram with the empirical means, and
        ii. Histogram with the diagonal and upper off-diagonal elements of the
            empirical covariance matrix.

Plots are saved as pdf's in a directory specified by DIRECTORY located in the
current working directory.  If the directory doesn't exists it will be created.
Plots currently stored in the directory will be overwritten.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from fd import sample
import utilities as utils

SEED = 1234
M, N, NREPS = 4, 1000, 100
TRUE_MEAN = np.linspace(0, 3, M)
TRUE_RHO = np.linspace(0, 0.9, 10)
DIRECTORY = "gauss_sample_plots"
NBINS = 50

def main():
    if not os.path.exists(DIRECTORY):
        os.mkdir(DIRECTORY)

    rng = np.random.default_rng(SEED)

    for rho in TRUE_RHO:

        emp_rho = utils.MatrixContainer(NREPS, M)
        emp_mean = utils.VectorContainer(NREPS, M)

        for n in range(NREPS):
            c = sample._construct_corr_matrix(M, rho)
            Y = sample.multivariate_gauss(TRUE_MEAN, c, N, seed=rng)

            emp_mean.append(np.mean(Y, 1))
            emp_rho.append(np.cov(Y))
    
        assert emp_rho._next == emp_rho._size

        rho_string = "100xrho_{}".format(int(100*rho))

        utils.hist(emp_rho.vals, (rho, 1), 
                    (r"$C_{ij}$ = "+ str(rho)+r" for $i \neq j$",
                     r"$C_{ii}$ = 1"),
                    "Covariance matrix values",
                    "{}/cov_{}.pdf".format(DIRECTORY, rho_string),
                    bins=NBINS)
        utils.hist(emp_mean.vals, TRUE_MEAN, 
                    [r"$\mu$ = " + str(m) for m in TRUE_MEAN],
                    "Mean values",
                    "{}/mean_{}.pdf".format(DIRECTORY, rho_string),
                    bins=NBINS)

if __name__ == "__main__":
    main()
