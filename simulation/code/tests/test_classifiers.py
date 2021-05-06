from unittest import TestCase, main

import numpy as np

from fd import classifiers as cl
from fd import sample, stats


class ClassifierTestHelper:
    """Mix in for subclasses of the FDBaseCl class.

    As there is no obvious way to validate the values that  
    compute_scores, compute_ranks, and infer_labels methods
    produce, consistency checks are performed. 

    This mixin class has tools for testing consistency between

        * compute_scores and compute_ranks
        * compute_scores and infer_labels

    subclassing requires that the following attributes be defined:

        1.  self.auc: ndarray of auc values.  For testing FD this will
            only have a single AUC entry.  For testing ensembles,
            the number of entries will match self.M
        2. self.cl: instance of classifier to be tested
        3. self.N: number of samples
        4. self.corr_coef: target conditional correlation between 
            base classifiers
        5. self.prevalence: fraction of samples from positive class
        6. self.Nreps: number of replicate simulation experiments to perform

    where 3-6 can be set by calling self._common_parameters() and method:

        * _simulate_and_fit: run simulation and fit classifier if necessary

    Subclassing can be performed as in the following example:
   
    class TestMyClassifier(ClassifierTestHelper, TestCase):
        def setUp(self):
            self._common_parameters()
            self.M = 10  # number of base classifiers to simulate
            self.auc = np.linspace(0.51, 0.9, self.M)
            self.cl = cl.MyFDClassifierClass()
    
        def _simulate_and_fit(self):
            self._update_sim_data()
            self.cl.fit(self.R, self.y)
    """
    def _common_parameters(self):
        self.N = 1000
        self.corr_coef = 0
        self.prevalence = 0.3
        self.Nreps = 100

    def _update_sim_data(self):
        """Update simulation data"""
        self.R, self.y = sample.data_set(self.auc,
                                        self.corr_coef, 
                                        self.prevalence, 
                                        self.N)

    def _simulate_and_fit(self):
        raise NotImplementedError

    def test_compute_ranks(self):
        """Verify that sample ranks follow convention.

        The rank convention is that relatively high sample scores
        correspond to low sample rank.  Consequently, for two samples
        i and j,

        rank_i < rank_j  implies that  score_i >= score_j

        with equality representing "ties".
        """
        for _ in range(self.Nreps):
            self._simulate_and_fit()

            scores = self.cl.compute_scores(self.R)
            ranks = self.cl.compute_ranks(self.R)
    
            # indexes corresponding to rank array sorted low to high
            sorted_idx = np.argsort(ranks)
    
            # step-wise test convention specified in docstring for all samples

            i = sorted_idx[0]
            for j in sorted_idx[1:]:
                self.assertTrue(ranks[i] < ranks[j])
                self.assertTrue(scores[i] >= scores[j])
    
                i = j

    def test_infer_labels(self):
        """Verify implementation of class label inference.

        A sample score > 0 corresponds to an inferred class label of 1,
        otherwise the class label is 0.
        """
        for _ in range(self.Nreps):
            self._simulate_and_fit()

            scores = self.cl.compute_scores(self.R)
            y_inferred = self.cl.infer_labels(self.R)
    
            for i, s in enumerate(scores):
                if s > 0:
                    self.assertEqual(y_inferred[i], 1)
                else:
                    self.assertEqual(y_inferred[i], 0)


class TestFDdataValidation(TestCase):
    """Test FD class methods for validating input data."""
    def setUp(self):
        self.N, self.prevalence = 500, 0.3
        self.r, self.y = sample.data_set(np.array([0.9]),
                0, self.prevalence, self.N)
        self.r = self.r.squeeze()

        self.cl = cl.FD()
        
    def test_validate_rank_input_vals(self):
        """Raise exception when input ranks are not in [1,N]"""
        with self.assertRaises(ValueError):
            self.r[10] = 0
            self.cl._validate_rank(self.r)

    def test_validate_rank_shape_requirement(self):
        """Raise exception if rank ndarray isn't 1-dimensional"""
        with self.assertRaises(ValueError):
            self.r = np.vstack([self.r, self.r])
            self.cl._validate_rank(self.r)

    def test_validate_data_label_val(self):
        """Raise exception when input labels are not in {0,1}"""
        with self.assertRaises(ValueError):
            self.y[10] = -1
            self.cl._validate_data(self.r, self.y)

    def test_validate_data_label_shape_requirement(self):
        """Raise exception if label ndarray isn't 1-dimensional"""
        with self.assertRaises(ValueError):
            self.y = np.vstack([self.y, self.y])
            self.cl._validate_data(self.r, self.y)

    def test_validate_data_rank_requirements(self):
        """Raise exception if _validate_data calls _validate_rank."""
        with self.assertRaises(ValueError):
            # want r.ndim == 2 while r.size = N
            # this is because a ValueError will be raised if r.size != y.size,
            # which is not the desired test
            self.r = self.r.reshape(2, int(self.N/2))
            assert self.r.size == self.y.size
            self.cl._validate_data(self.r, self.y)

    def test_validate_data_rank_label_shape_mismatch(self):
        """Raise exception if shape of rank and label don't match."""
        with self.assertRaises(ValueError):
            self.cl._validate_data(self.r[:-1], self.y)

    def test_validate_N(self):
        """Raise exception if N isn't an int > 0."""
        with self.assertRaises(ValueError):
            self.cl._validate_N([3,2])
        with self.assertRaises(ValueError):
            self.cl._validate_N(np.array([4,3]))
        with self.assertRaises(ValueError):
            self.cl._validate_N(0)
        with self.assertRaises(ValueError):
            self.cl._validate_N(3.2)


class TestFDValidationCallsByMethods(TestCase):
    """Ad hoc validation that appropriate data validation method is called."""
    def setUp(self):
        self.N, self.prevalence = 500, 0.3
        self.r, self.y = sample.data_set(np.array([0.9]),
                0, self.prevalence, self.N)
        self.r = self.r.squeeze()

        self.cl = cl.FD()

    def test_get_beta_input_val(self):
        """Raise exception if invalid input."""
        with self.assertRaises(ValueError):
            self.cl.get_beta(0)

    def test_get_mu_input_val(self):
        """Raise exception if invalid input."""
        with self.assertRaises(ValueError):
            self.cl.get_mu(0)

    def test_get_thresh_input_val(self):
        """Raise exception if invalid input."""
        with self.assertRaises(ValueError):
            self.cl.get_thresh(0)

    def test_fit_input_val(self):
        """Raise exception if invalid input."""
        with self.assertRaises(ValueError):
            self.cl.fit(self.y, self.y)

        with self.assertRaises(ValueError):
            self.cl.fit(self.r, self.r)

    def test_compute_scores_input_val(self):
        """Raise exception if invalid input."""
        with self.assertRaises(ValueError):
            self.r[10] = self.r[14]
            self.cl.compute_scores(self.r)

    def test_compute_ranks_input_val(self):
        """Raise exception if invalid input."""
        with self.assertRaises(ValueError):
            self.r[10] = self.r[14]
            self.cl.compute_ranks(self.r)

    def test_infer_labels_input_val(self):
        """Raise exception if invalid input."""
        with self.assertRaises(ValueError):
            self.r[10] = self.r[14]
            self.cl.infer_labels(self.r)


class TestFDparameterRetrieval(TestCase):
    """Verify computation, retrieval, and update of parameter values.

    Retrieving parameters from the FD class involves calculations using
    the two inferred parameters from data, namely beta' and the positive
    class prevalence.  

    In these tests, I specifically set the values of inferred parameters,
    and verify that the set and dependent parameters match expectations.  These
    are imperfect tests, as the expected parameter values are computed from the
    same equations implemented in the FD class.  However, the hope is that
    an independent implementation may provide some assurance to their accuracy.
    """
    def setUp(self):
        """Initialize pars and random number generators and FD class."""
        # Parameter values to conduct tests over
        self.prevalences = (i/10 for i in range(1, 10))
        self.Nsamples = (10*i for i in range(1, 5))

        self.rng = np.random.default_rng()

        self.cl = cl.FD()

        # True FD parameters for validating FD parameter retieval
        self._beta_p = None
        self._prevalence = None

    def _update_pars(self, p):
        """Update FD model parameters.

        Updates current model prevalence to the specified p, and model
        beta_p value by randomly sampling from N(0,1).

        Args:
            p: (float) positive class prevalences on (0,1)
        """
        self._prevalence = p
        self._beta_p = self.rng.normal()

        self.cl.prevalence = p
        self.cl._beta_p = self._beta_p

    def _get_mu_p(self):
        """Compute mu_p to compare to FD class implementation."""
        mp = np.log(np.sinh(self._beta_p*(1-self._prevalence)/2) / 
                    np.sinh(self._beta_p*self._prevalence/2))
        mp /= self._beta_p
        return 0.5 - mp

    def _get_beta(self, n):
        """Compute beta to compare to FD class implementation."""
        return self._beta_p / n

    def _get_mu(self, n):
        """Compute mu to compare to FD class implementation."""
        return self._get_mu_p() * n

    def _get_thresh(self, n):
        """Compute threshold rank to compare to FD class implementation."""
        t = self._get_mu_p() * n
        t += n * np.log((1-self._prevalence) / self._prevalence) / self._beta_p
        return t
 
    def test_beta(self):
        """Confirm value returned by get_beta."""
        for p in self.prevalences:
            for n in self.Nsamples:
                self._update_pars(p)
                
                self.assertEqual(self.cl.get_beta(n), self._get_beta(n))

    def test_mu(self):
        """Confirm value returned by get_mu."""
        for p in self.prevalences:
            for n in self.Nsamples:
                self._update_pars(p)
                
                self.assertAlmostEqual(self.cl.get_mu(n), 
                                       self._get_mu(n), 
                                       places=10)

    def test_thresh(self):
        """Confirm value returned by get_thresh."""
        for p in self.prevalences:
            for n in self.Nsamples:
                self._update_pars(p)

                self.assertAlmostEqual(self.cl.get_thresh(n), 
                                       self._get_thresh(n),
                                       places=10)


class TestFDfit(TestCase):
    """Verification of ad hoc relationships between inputs and inferred parameters.

    Test fit by verifying ad hoc relationships between AUC, prevalence and FD
    parameters beta and mu are satisfied.
    """
    def setUp(self):
        """Set default parameters and init FD class."""
        self.N = 1000
        self.Nreps = 100
        self.corr_coef = 0

        self.auc = 0.8
        self.prevalence = 0.3

        self.cl = cl.FD()

    def _simulate_and_fit(self, auc=None, prevalence=None):
        """Generate simulation data and fit FD model.

        Given either default parameters stored as TestFDfit class attributes,
        or those provided, generate simulation data.  Using FD class instance
        bound to TestFDfit class attribute self.cl, fit the data.  Test are
        conducted by accessing the trained FD class instance.
        """
        if auc is None:
            auc = np.array([self.auc])
        elif not isinstance(auc, np.ndarray):
            auc = np.array([auc])

        if prevalence is None:
            prevalence = self.prevalence

        r, y = sample.data_set(auc, self.corr_coef, prevalence, self.N)
        r = r.squeeze()

        self.cl.fit(r, y)

    def test_relative_beta_p_lessthan_half(self):
        """If AUC_i < AUC_j and prevalence < 1/2, then beta_i < beta_j."""
        auc_vals = np.array([0.1, 0.4, 0.6, 0.9])

        for _ in range(self.Nreps):

            self._simulate_and_fit(auc=auc_vals[0])
            beta_prev = self.cl.get_beta(self.N)

            for auc in auc_vals[1:]:
                self._simulate_and_fit(auc=auc)

                self.assertTrue(beta_prev < self.cl.get_beta(self.N))

                beta_prev = self.cl.get_beta(self.N)

    def test_relative_beta_p_greaterthan_half(self):
        """If AUC_i < AUC_j and prevalence > 1/2, then beta_i < beta_j."""
        auc_vals = np.array([0.1, 0.4, 0.6, 0.9])
        prevalence = 0.8

        for _ in range(self.Nreps):

            self._simulate_and_fit(auc=auc_vals[0], 
                                   prevalence=prevalence)
            beta_prev = self.cl.get_beta(self.N)

            for auc in auc_vals[1:]:
                self._simulate_and_fit(auc=auc, prevalence=prevalence)

                self.assertTrue(beta_prev < self.cl.get_beta(self.N))

                beta_prev = self.cl.get_beta(self.N)

    def test_negative_beta_p_lessthan_half(self):
        """If AUC < 0.5 and prevalence < 1/2, then beta < 0."""
        auc = 0.2
        for _ in range(self.Nreps):
            self._simulate_and_fit(auc=auc)
            self.assertTrue(self.cl.get_beta(self.N) < 0)

    def test_negative_beta_p_greaterthan_half(self):
        """If AUC < 0.5 and prevalence > 1/2, then beta < 0."""
        auc = 0.2
        prevalence = 0.8
        for _ in range(self.Nreps):
            self._simulate_and_fit(auc=auc, prevalence=prevalence)
            self.assertTrue(self.cl.get_beta(self.N) < 0)

    def test_positive_beta(self):
        """If AUC > 0.5 and p < 1/2, beta > 0."""
        for _ in range(self.Nreps):
            self._simulate_and_fit()
            self.assertTrue(self.cl.get_beta(self.N) > 0)

    def test_relative_mu_auc_lessthan_half(self):
        """If AUC < 1/2 and prevalence_i < prevalence_j, then mu_i > mu_j."""
        auc = 0.2
        prevalences = np.linspace(0.1, 0.9, 9)
        for _ in range(self.Nreps):

            self._simulate_and_fit(prevalence=prevalences[0], auc=auc)
            prev_mu = self.cl.get_mu(self.N)
           
            for p in prevalences[1:]: 
                self._simulate_and_fit(prevalence=p, auc=auc)

                self.assertTrue(prev_mu > self.cl.get_mu(self.N))

                prev_mu = self.cl.get_mu(self.N)

    def test_relative_mu_auc_greaterthan_half(self):
        """If AUC > 1/2 and prevalence_i < prevalence_j, then mu_i < mu_j."""
        prevalences = np.linspace(0.1, 0.9, 9)
        for _ in range(self.Nreps):

            self._simulate_and_fit(prevalence=prevalences[0])
            prev_mu = self.cl.get_mu(self.N)
           
            for p in prevalences[1:]: 
                self._simulate_and_fit(prevalence=p)

                self.assertTrue(prev_mu < self.cl.get_mu(self.N))

                prev_mu = self.cl.get_mu(self.N)


class TestFDclassifier(ClassifierTestHelper, TestCase):
    """Test classification methods of the FD classifier.

    Using the tests defined in ClassifierTestHelper perform consistency
    checks between:

    * compute_scores and compute_ranks
    * compute_scores and infer_labels
    """
    def setUp(self):
        self._common_parameters()
        self.auc = np.array([0.8])
        self.cl = cl.FD()

    def _simulate_and_fit(self, auc=None):
        if auc is not None:
            self.auc = np.array([auc])
        self._update_sim_data()
        self.R = self.R.squeeze()

        self.cl.fit(self.R, self.y)

    def test_classifier_determined_ranks(self):
        """Verify that classifier ranks match input ranks.

        The FD base classifier produces scores that are monotonic
        with the input sample ranks.  If

        1. AUC > 1/2 classifier ranks = input ranks
        2. AUC < 1/2 classifier ranks = N+1 - input ranks

        the reason for 2. methods with AUC < 1/2 in effect have adopted
        a convention that is opposite to the stated FD convention.
        """
        for _ in range(self.Nreps):
            for auc in np.linspace(0.1, 0.9, 15):

                self._simulate_and_fit(auc=auc)

                if stats.rank_2_auc(self.R, self.y) > 0.5:
                    test_array = self.R == self.cl.compute_ranks(self.R)
                elif stats.rank_2_auc(self.R, self.y) < 0.5:
                    r = self.N+1 - self.cl.compute_ranks(self.R) 
                    test_array = self.R == r

                self.assertTrue(test_array.all())


class TestFDensemble(ClassifierTestHelper, TestCase):
    """Test classification methods of the FDensemble classifier.

    Verify that compute_scores method validates input rank data.  Using 
    the tests defined in ClassifierTestHelper perform consistency
    checks between:

    * compute_scores and compute_ranks
    * compute_scores and infer_labels
    """
    def setUp(self):
        self._common_parameters()
        self.M = 10
        self.auc = np.linspace(0.55, 0.9, self.M)
        self.cl = cl.FDensemble()

    def _simulate_and_fit(self):
        self._update_sim_data()
        self.cl.fit(self.R, self.y)

    def test_compute_scores_data_validation(self):
        """Verify that compute_scores validates input rank data."""

        self._simulate_and_fit()
        
        with self.assertRaises(ValueError):
            self.cl.compute_scores(self.R[0, :])

        with self.assertRaises(ValueError):
            self.cl.compute_scores(self.R[:-1,:])

        with self.assertRaises(AttributeError):
            self.cl.compute_scores(self.R.tolist())

        with self.assertRaises(ValueError):
            self.R[0,4] = self.R[0, -1]
            self.cl.compute_scores(self.R)
 

class TestWoc(ClassifierTestHelper, TestCase):
    """Test classification methods of the Woc classifier.

    Verify that compute_scores method validates input rank data, and
    that calling the fit method raises NotImplementedError.  Using
    the tests defined in ClassifierTestHelper perform consistency
    checks between:

    * compute_scores and compute_ranks
    * compute_scores and infer_labels
    """
    def setUp(self):
        self._common_parameters()
        self.M = 10
        self.auc = np.linspace(0.55, 0.9, self.M)
        self.cl = cl.Woc()

    def _simulate_and_fit(self):
        self._update_sim_data()

    def test_fit_exception_is_raised(self):
        """Verify fit method raises NotImplementedError."""

        self._simulate_and_fit()

        with self.assertRaises(NotImplementedError):
            self.cl.fit()
        with self.assertRaises(NotImplementedError):
            self.cl.fit(1,3, self.R)

    def test_compute_scores_data_validation(self):
        """Verify that compute_scores validates input rank data."""

        self._simulate_and_fit()
        
        with self.assertRaises(ValueError):
            self.cl.compute_scores(self.R[0, :])

        with self.assertRaises(AttributeError):
            self.cl.compute_scores(self.R.tolist())

        with self.assertRaises(ValueError):
            self.R[0,4] = self.R[0, -1]
            self.cl.compute_scores(self.R)
                

class TestBestInd(ClassifierTestHelper, TestCase):
    def setUp(self):
        self._common_parameters()

        self.M = 4 
        self.auc = np.linspace(0.51, 0.9, self.M)
        self.cl = cl.BestInd()

        self.nreps = 10

    def _simulate_and_fit(self):
        self._update_sim_data()
        self.cl.fit(self.R, self.y)

    def test_compute_scores_data_validation(self):
        """Verify that rank data validation is called."""
        self._simulate_and_fit()

        # 1-d data input as opposed to 2-d
        with self.assertRaises(ValueError):
            self.cl.compute_scores(self.R[0, :])

        # one row's rank data has ties
        with self.assertRaises(ValueError):
            self.R[2,7] = self.R[2, 55]
            self.cl.compute_scores(self.R)

    def test_fit_rank_data_validation(self):
        """Veify that rank data are validated prior to fit."""
        self._simulate_and_fit()

        with self.assertRaises(ValueError):
            self.R[1,3] = self.R[1, 4]
            self.cl.fit(self.R, self.y)

    def test_fit_label_data_validation(self):
        """Verify that label data are validated prior to fit."""
        self._simulate_and_fit()

        with self.assertRaises(ValueError):
            self.y[10] = -1
            self.cl.fit(self.R, self.y)

    def test_fit_data_validation(self):
        """Verify that mismatch in R.shape[1] and y.size raises exception."""
        self._simulate_and_fit()

        with self.assertRaises(ValueError):
            self.cl.fit(self.R[:, :(self.y.size-2)], self.y)

    def test_fit(self):
        """Verify index of best performing method.

        AUC values are in ascending order, so the the index of the 
        best base classifier must be the largest possible index, i.e. M-1.
        """
        for _ in range(self.nreps):
            self._simulate_and_fit()
            self.assertEqual(self.cl._best_idx, self.M-1)
        



if __name__ == "__main__":
    main()
