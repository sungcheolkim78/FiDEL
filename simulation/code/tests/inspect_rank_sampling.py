"""Qualitative inspection of rank AUC and correlation coefficients

Running this program will:

    1. Generate NREPS replicate data sets of N sample rank predictions by M
        base classifiers, and N sample class labels.
    2. For different conditional correlation coefficients, print two plots:
        i. Histogram with the empirical auc, and
        ii. Histogram with the diagonal and upper off-diagonal, elements of the
            empirical class conditioned correlation matrices.

Plots are saved as pdf's in a directory specified by DIRECTORY located in the
current working directory.  If the directory doesn't exists it will be created.
Plots currently stored in the directory will be overwritten.
"""

import os
import numpy as np
from fd import sample
from fd import stats
import utilities as utils

SEED = 1234
M, N, NREPS = 5, 1000, 100
PREVALENCE = 0.3
TRUE_AUC = np.linspace(0.5, 0.9, M)
TRUE_RHO = np.linspace(0, 0.9, 4)
DIRECTORY = "rank_sample_plots"
NBINS = 50

def main():
    if not os.path.exists(DIRECTORY):
        os.mkdir(DIRECTORY)

    auc = np.zeros(M)

    # set seed for pseudo random number generation
    rng = np.random.default_rng(SEED)

    for rho in TRUE_RHO:
        emp_rho_p = utils.MatrixContainer(NREPS, M)  # positive class corrs
        emp_rho_n = utils.MatrixContainer(NREPS, M)  # negative class corrs
        emp_auc = utils.VectorContainer(NREPS, M)

        for n in range(NREPS):
            R, y = sample.data_set(TRUE_AUC, rho, PREVALENCE, N, seed=rng)
            
            for i in range(M):
                auc[i] = stats.rank_2_auc(R[i, :], y)

            emp_auc.append(auc)
            emp_rho_p.append(np.corrcoef(R[:, y == 1]))
            emp_rho_n.append(np.corrcoef(R[:, y == 0]))

        rho_string = "100xrho_{}".format(int(100*rho))

        utils.hist(np.hstack([emp_rho_p.vals, emp_rho_n.vals]),
                    (rho, 1), 
                    (r"$C_{ij}$ = "+str(rho)+r"for $i \neq j$", 
                     r"$C_{ii}$ = 1"),
                    "Conditional correlation coefficients",
                    "{}/corr_{}.pdf".format(DIRECTORY, rho_string),
                     bins=NBINS)
        utils.hist(emp_auc.vals, 
                    TRUE_AUC, 
                    [r"AUC = " + str(auc) for auc in TRUE_AUC],
                    "AUC",
                    "{}/auc_{}.pdf".format(DIRECTORY, rho_string),
                    bins = NBINS)

    return 0


if __name__ == "__main__":
    main()
