"""Implementations of rank based binary classifiers

Available Classes:
- BestInd: Determines best individual base classifier, and applies it to
    input data.
- FD: Implementation of the Fermi-Dirac distribution for binary classification
- FDensemble: Implementation of an ensemble of Fermi-Dirac binary classifiers
- Woc: Implementation of the Wisdom of Crowds classifier
"""

import numpy as np
from scipy.optimize import newton
from scipy.stats import rankdata

from . import stats
from . import validate as val


class FDBaseCl:
    """Abstract base class for all classifiers.

    Class is not meant to be called directly, but subclassed by specific
    classifier implementations.  Subclasses must define methods enumerated
    in "Subclass Methods", and automatically have available those in 
    "Available Methods".

    Subclass Methods:
    - fit: infer model parameters from sample ranks and class label data.
    - compute_scores: compute model defined sample scores from sample rank data.

    Available Methods:
    - compute_ranks: classifier determined sample ranks
    - infer_labels: classifier inferred sample class labels
    """
    def fit(self, *_):
        """Train model given data."""
        raise NotImplementedError

    def compute_scores(self, *_):
        """Given a trained (if necessary) instance, compute sample scores.

        Convention for scores is that a sample from the positive class should
        have a relatively higher score than one from the negative class.
        """
        raise NotImplementedError

    def compute_ranks(self, R):
        """Compute sample rank according to classifier

        Assign rank so that samples with relatively high scores are assigned 
        low rank.  Samples with identical scores are assigned rank ordinally.

        Args:
            R: ((N,) ndarray or (M, N) ndarray) independent rows of sample ranks, 
                no ties in row

        Returns:
            ((N,) ndarray) of sample ranks
        """
        return rankdata(-self.compute_scores(R), method="ordinal")

    def infer_labels(self, R):
        """Infer sample class labels.

        Sample class labels are inferred from classifier scores.  These scores are
        computed using the input rank data.  Samples with a score > 0 are assigned 
        a positive (1) class label, while samples with a score <= 0 are assigned a
        negative (0) class label.

        Args:
            R: ((N,) ndarray or (M, N) ndarray) independent rows of sample ranks, 
                no ties in row

        Returns:
            y: ((N,) ndarray) inferred binary [0,1] sample class labels
        """
        s = self.compute_scores(R)
        y = np.zeros(s.size)
        y[s > 0] = 1
        return y


class BestInd(FDBaseCl):
    """The best individual method classifier.

    Class is to apply FD convention to finding and applying the classifier
    with the highest training set AUC.

    Available Methods:
    - fit: find index corresponding to the classifier with the highest 
        AUC on training data set
    - compute_scores: (N+1)/2 - (sample ranks by best individual classifier).

    Inherited Methods see FDBaseCl for documentation:
    - compute_ranks
    - infer_labels
    """
    def __init__(self):
        self._best_idx = None

    def _validate_rank(self, R):
        """Raise Exception if R not ((M, N) ndarray) of rank values."""
        if R.ndim != 2:
            raise ValueError("Rank data must be a 2-d ndarray")
        val.validate_rank_data(R)

    def _validate_data(self, R, y):
        """Raise exception if R and y are not valid."""
        self._validate_rank(R)
        val.validate_label_data(y)
        if R.shape[1] != y.size:
            raise ValueError("Number of columns of rank array must equal the "
                            "number of elements in the label array.")

    def fit(self, R, y):
        """Given sample rank and label data, compute best AUC.
    
        Compute the AUC given each base classifier's sample rank predictions (rows),
        and true class labels.  Return the maximum AUC value.
    
        Args:
            R: ((M, N) ndarray) independent rows of sample ranks, no ties in row
            y: ((N,) ndarray) of sample class labels [0, 1].
    
        Returns:
            best_ind_auc: (float) The highest empirical AUC over base classifier
                sample rank predictions.
        """
        self._validate_data(R, y)

        self._best_idx = None
        best_ind_auc = 0 

        for i in range(R.shape[0]):
    
            tmp_auc = stats.rank_2_auc(R[i, :], y)
    
            if tmp_auc > best_ind_auc:
                best_ind_auc = tmp_auc
                self._best_idx = i
    
    def compute_scores(self, R):
        """Compute scores.

        Args:
            R: ((M, N) ndarray) independent rows of sample ranks, no ties in row
        """
        self._validate_rank(R)
        N = R.shape[1]
        return (N+1)/2 - R[self._best_idx, :]


class FD(FDBaseCl):
    """Fermi-Dirac distribution for binary classification.

    Available Methods:
    - fit: infer beta' and positive class prevalence from sample rank and 
        class label data
    - get_beta: given N samples retrieve inferred beta
    - get_mu: given N samples retrieve inferred mu
    - get_thresh: given N samples retrieve rank threshold for class 
        label inference
    - compute_scores: compute log-likelihood ratio for each sample

    Inherited Methods see FDBaseCl for documentation:
    - compute_ranks
    - infer_labels

    References:
        S. Kim et al. "Learning from Fermions: the Fermi-Dirac Distribution
            Provides a Calibrated Probabilistic Output for Binary Classifiers."
            in review (2021)
    """
    def __init__(self):
        self.prevalence = None
        self._beta_p = None

    @staticmethod
    def _compute_mu_p(beta_p, prevalence):
        """Compute \mu'.

        \mu' = 1/2 - (1/b') ln( sinh( b' (1-p) / 2) / sinh( b' p / 2) )

        with b' representing \beta', ln the natural logarithm, and p the
        positive sample class prevalence.

        Args:
            beta_p: (float) model parameter
            prevalence: (float) number of positive class / number samples (0, 1)

        Returns:
            (float) \mu'
        """
        log_term = np.log(
                        np.sinh(0.5*beta_p*(1-prevalence)) / 
                        np.sinh(0.5*beta_p*prevalence)
                        )
        return 0.5 - log_term / beta_p

    @staticmethod
    def _find_beta(beta, r, prevalence, N, N1, c):
        """Equation of constraint corresponding to \beta in max-ent dist.

        Equation is equivalent to that in S. Kim et al. and is used in 
        conjunction with scipy.optimize.newton for computing \beta from data.
        The equation of constraint is as follows:

        E[R | Y=1] = (N+1) / 2 - (1 - prevalence) D

        where D represents delta := E[R | Y=0] - E[R | Y=1], and the conditional
        probability for computing the expectation is:
        
        Pr(R=r | Y=1) = Pr(Y=1 | R=r) Pr(R=r) / Pr(Y=1) = Pr(Y=1 | R=r) / N1, and
        Pr(Y=1 | R=r) = (1 + exp(beta(r - mu)))^{-1}.

        where mu is computed from beta, N, and prevalence of the positive class
        samples.

        Args:
            beta: (float)
            r: ((N,) ndarray) sample ranks, no ties
            prevalence: (float) number of positive class / number samples (0, 1)
            N: (int) corresponding to max(r)
            N1: (float) prevalence * N
            c: (float)  R.H.S. of the equation of constraint described above

        Returns:
            (float)
        """
        mu = N * FD._compute_mu_p(N*beta, prevalence)
        inv_prob = 1 + np.exp(beta*(r - mu))
        return np.sum(r / inv_prob) / N1 - c

    def _validate_N(self, N):
        """Raise ValueError if N not an integer > 0"""
        if np.mod(N,1) != 0 or N < 1:
            raise ValueError("N must be integer greater than 0.")

    def _validate_rank(self, r):
        """Raise Exception if r not ((N,) ndarray) of rank values."""
        if r.ndim != 1:
            raise ValueError("Rank data must be a 1-d ndarray")
        val.validate_rank_data(r)

    def _validate_data(self, r, y):
        """Raise exception if r and y are not valid."""
        self._validate_rank(r)
        val.validate_label_data(y)
        if r.size != y.size:
            raise ValueError("Rank and label arrays must have same size.")

    def fit(self, r, y):
        """Fit FD model parameters from data.

        Args:
            r: ((N,) ndarray) sample ranks, no ties
            y: ((N,) ndarray) of sample class labels [0, 1].
        """
        self._validate_data(r, y)

        N = r.size
        delta = stats.delta(r, y)

        self.prevalence = np.mean(y)

        # I believe the default method is the secant method for
        # finding root.  This is because the derivative of 
        # self._find_beta is not provided
        beta = newton(self._find_beta, 
                      12 * delta / N**2, 
                      args = (r, self.prevalence, 
                              N, self.prevalence*N, 
                              0.5*(N+1) - (1-self.prevalence)*delta)) 
        self._beta_p = beta * N

    def get_beta(self, N):
        """Given N and beta' return beta.
    
        Args:
            N: (int) number of samples.

        Returns:
            (float) beta
        """
        self._validate_N(N)
        return self._beta_p / N

    def get_mu(self, N):
        """Given N and mu' return mu.

        Args:
            N: (int) number of samples.

        Returns:
            (float) mu
        """
        self._validate_N(N)
        return N * self._compute_mu_p(self._beta_p, self.prevalence)

    def get_thresh(self, N):
        """Given FD parameters compute the threshold rank for class inference.

        Args:
            N: (int) number of samples.

        Returns:
            (float): rank threshold for inferring class labels
        """
        return (self.get_mu(N) + 
            np.log((1-self.prevalence) / self.prevalence) / self.get_beta(N))

    def compute_scores(self, r):
        """Return the log-likelihood ratio.

        Args:
            r: ((N,) ndarray) sample ranks, no ties

        Returns:
            ((N,) ndarray) of scores
        """
        self._validate_rank(r)
        N = r.size
        return self.get_beta(N) * (self.get_thresh(N) - r)


class FDensemble(FDBaseCl):
    """Ensemble of Fermi-Dirac binary classifiers.

    Available Methods:
    - fit: infer beta' for each base classifier and positive class prevalence
        from sample rank and class label data
    - compute_scores: compute log-likelihood ratio for each sample

    Inherited Methods see FDBaseCl for documentation:
    - compute_ranks
    - infer_labels

    References:
        S. Kim et al. "Learning from Fermions: the Fermi-Dirac Distribution
            Provides a Calibrated Probabilistic Output for Binary Classifiers."
            in review (2021)
    """
    def __init__(self):
        self._M = None
        self._base_cls = None

    def fit(self, R, y):
        """Fit FD distribution for each base classifier's rank predictions.

        Args:
            R: ((M, N) ndarray) independent rows of sample ranks, no ties in row
            y: ((N,) ndarray) of sample class labels [0, 1].
        """
        if R.ndim != 2:
            raise ValueError("Input ranks must be a 2-d ndarray.")

        self._M = R.shape[0]
        self._base_cls = [FD() for _ in range(self._M)]

        for i, base_cl in enumerate(self._base_cls):
            base_cl.fit(R[i, :], y)

    def compute_scores(self, R):
        """Compute FD ensemble score for each sample.

        For each sample k \in {1, 2, ..., N} a sample score, denoted s_k, is 
        computed by adding the log-likelihood ratio from each of the M trained
        base classifiers.  The kth sample score is then

        s_k = \sum_{i=1}^M  \beta_i (r_i^* - r_{ik})

        with r_{ik} being the kth sample rank prediction by the ith base 
        classifier.

        Args:
            R: ((M, N) ndarray) independent rows of sample ranks, no ties in row

        Returns:
            s: ((N,) ndarray) of sample ensemble scores.
        """
        if R.ndim != 2 or R.shape[0] != self._M:
            raise ValueError("Input rank data must be 2-d array" 
                            "with {} rows".format(self._M))

        s = np.zeros(R.shape[1])

        for i, base_cl in enumerate(self._base_cls):
            s += base_cl.compute_scores(R[i, :])

        return s
    

class Woc(FDBaseCl):
    """The Wisdom of Crowds classifier.

    References:
        D. Marbach et al. "Wisdom of crowds for robust gene network inference."
            Nature Methods 9.8 (2012)
    """
    def _validate_rank(self, R):
        """Raise Exception if R not ((M, N) ndarray) of rank values."""
        if R.ndim != 2:
            raise ValueError("Rank data must be a 2-d ndarray")
        val.validate_rank_data(R)

    def compute_scores(self, R):
        """Compute Woc sample scores.

        The Wisdom of Crowds score for sample k is computed as,

        s_k = (N+1) / 2 - (1/M) \sum_{i=1}^M r_{ik}

        where r_{ik} is the kth sample rank prediction by base classifier i,
        N is the number of samples, and M is the total number of base
        classifiers.

        Args:
            R: ((M, N) ndarray) independent rows of sample ranks, no ties in row

        Returns:
            ((N,) ndarray) of sample scores, a low score corresponds to 
                a sample believed to be from the positive class
        """
        self._validate_rank(R)
        N = R.shape[1]
        return (N+1)/2 - np.mean(R, 0)




