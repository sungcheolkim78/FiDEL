"""Sample rank data sets from Gaussian distributions.

This module implements Gustavo's prescription for generating synthetic
data.  The data consists of a (M, N) ndarray, R, of N sample rank predictions
by M base classifiers and (N,) ndarray of true sample labels.  The synthetic
rank predictions may be correlated by specifying a correlation coefficient.

Available Functions:
- data_set: generate a synthetic data set composed of sample 
    ranks and class labels
- multivariate_gauss: generate samples from the multivariate Gaussian 
    distribution
"""

import numpy as np
from scipy.special import ndtri     # inverse standard normal cumulative
from scipy.stats import rankdata


def _construct_corr_matrix(M, rho):
    """Construct correlation matrix.

    Construct a correlation matrix in which 

    C_{ij} = rho for all i \neq j.

    Args:
        M: (int) > 0, representing the number of rows and columns
        rho: (float) on interval [0, 1) representing the correlation coefficient

    Returns:
        ((M, M) ndarray) correlation matrix
    """
    if rho < 0 or rho >= 1:
        raise ValueError("The correlation coefficient (rho)"
                    " is defined on interval [0,1).")
    elif M < 1:
        raise ValueError("Required that M > 1.")

    c = rho + np.zeros(shape=(M, M))
    for i in range(M):
        c[i, i] = 1
    return c

def multivariate_gauss(m, cov, N, seed=None):
    """Sample from multivariate Gaussian distribution.

    Algorithm designed by Gustavo Stolovitzky

    Args:
        m: ((M,) ndarray) M > 0, of means
        cov: ((M,M) ndarray) M > 0, covariance matrix
        N: (int) > 1, number of samples draw
        seed: seed value for np.random.default_rng, default is None,
            under default value (None) a seed is produced by the OS

    Returns:
        X: ((M, N) ndarray) of sampled Gaussian scores
    """
    M = m.size
    if m.ndim != 1:
        raise ValueError("m must be a 1-d ndarray of means")
    elif cov.shape != (M, M):
        raise ValueError("cov must have shape (m.size, m.size)")
    elif N < 1:
        raise ValueError("Required that N >= 1.")
    elif (cov != cov.transpose()).any():
        raise ValueError("Covariance matrix must be symmetric")

    # sample from N(0, 1)
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(M, N))

    # l (M,) ndarray of eigenvalues, 
    # v ((M,M) ndarray) of column eigenvectors where v[:, i] corresponds to l[i]
    l, v = np.linalg.eigh(cov)
    l = np.diag(np.sqrt(l))

    m = np.tile(m.reshape(M,1), (1, N))
    y = np.dot(v, np.dot(l, x))
    return y + m
    
def _auc_2_delta(auc, v):
    """Compute the difference of class conditioned means (delta) from the AUC.

    According to Marzban, reference below, delta is related to the AUC by

    delta = \sqrt{\sigma_0^2 + \sigma_1^2} \Phi^{-1} (AUC)

    with \sigma_y^2 begin the conditional variance given y and \Phi the standard
    normal cumulative distribution.

    Args:
        auc: (float) [0, 1]
        v: ((2) tuple) of (\sigma_0^2, \sigma_1^2)

    Returns:
        (float) E[s|y = 0] - E[s|y = 1]

    Reference:
        Marzban, "The ROC Curve and the Area under It as Performance Measures",
            Weather and Forecasting, 2004.
    """ 
    if auc < 0 or auc > 1:
        raise ValueError("AUC is defined on interval [0,1].")
    if len(v) != 2:
        raise ValueError(("Must supply len 2 tuple with class conditioned "
                        "variances"))
    if v[0] < 0 or v[1] < 0:
        raise ValueError("By definition, variances must be greater than 0.")
    return np.sqrt(v[0] + v[1]) * ndtri(auc)

def data_set(auc, corr_coef, prevalence, N, seed=None):
    """Sample rank data and sample class labels.

    Rank data are produced by rank ordering samples drawn from two Gaussian
    distributions.  Each Gaussian is representative of samples drawn from one
    of the two sample classes, and have unit variance and correlation specified by
    corr_coef.  The distance between Gaussians are determined by their respective
    means, which are computed from the specified AUC.

    Two samples with identical scores are ordinally assigned a rank value so that
    no two samples have identical rank.

    Args:
        auc: ((M,) ndarray) of auc values on the interval [0, 1]
        corr_coef: (float) correlation between classifier predictions [0, 1)
        prevalence: (float) number of positive class / number samples (0, 1)
        N: (int) > 1
        seed: any seed compatible with np.random.default_rng
    
    Returns:
        R: ((M, N) ndarray) independent rows of sample ranks, no ties in row
        y: ((N,) ndarray) binary [0,1] sample class labels
    """
    if isinstance(auc, float):
        auc = [auc]
    if prevalence <= 0 or prevalence >= 1:
        raise ValueError("Prevalence must by in interval (0,1).")

    # stats for sampling from multivariate Gaussian
    M = len(auc)
    N1 = int(N * prevalence)

    c = _construct_corr_matrix(M, corr_coef)
    delta = np.zeros(M)

    for i, auc_val in enumerate(auc):
        delta[i] = _auc_2_delta(auc_val, (c[i, i], c[i, i]))

    # create random number generator object accoring to seed
    rng = np.random.default_rng(seed)

    # sample from multivariate Gaussians
    s = np.hstack([multivariate_gauss(np.zeros(M), c, N1, seed=rng), 
                    multivariate_gauss(delta, c, N-N1, seed=rng)])

    # Construct the label array
    y = np.zeros(N)
    y[:N1] = 1

    # Construct the rank data array
    R = np.zeros(shape=(M, N))
    for i in range(M):
        R[i, :] = rankdata(s[i, :], method="ordinal")

    return R, y
