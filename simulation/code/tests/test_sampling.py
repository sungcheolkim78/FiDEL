from unittest import TestCase, main
import numpy as np
from fd import sample


class TestConstructCorrMatrix(TestCase):
    """Test _construct_corr_matrix output and raised exceptions."""
    def test_output_properties(self):
        """Verify output is as expected.

        Test 1: test output attributes:
            is an ndarray, ndim == 2, and correct shape
        Test 2: test output values:
            off-diagonal components == rho
            diagonal components == 1
        """
        M = 10
        rho = 0.1
        c = sample._construct_corr_matrix(M, rho)

        # Test 1: output attributes
        self.assertTrue(isinstance(c, np.ndarray))
        self.assertEqual(c.ndim, 2)
        self.assertTrue(c.shape == (M, M))

        # Test 2: output values
        # test elements of matrix
        for i in range(0, M):
            for j in range(0, M):
                if i == j:
                    self.assertEqual(c[i, i], 1)
                else:
                    self.assertEqual(c[i, j], rho)

    def test_eigendecomposition(self):
        """Verify Eigendecomposition of correlation matrix C.

        This tests applies Gustavo's sanity check of the Eigendecomposition 
        of the (M, M) correlation matrix C, in which:

        C_{ij} = \rho for all i \neq j.

        The Eigendecomposition of C results in Eigenvalues:
            a) M-1 Eigenvalues equal to (1-\rho)
            b) 1 Eigenvalue equal to 1 + (M-1) \rho
        and the Eigenvector of the unique Eigenvalue in (b) being
            v = [1, 1, ..., 1] / \sqrt{M}

        """
        M = 10
        rho = 0.1
        c = sample._construct_corr_matrix(M, rho)
        tolerance = 10

        # l is an ((M,) ndarray) eigenvalues in ascending order
        # v is an ((M, M) ndarray) of eigenvectors.  Each column vector
        # v[:, i] of v is the eigenvector corresponding to l[i]
        l, v = np.linalg.eigh(c)

        # verify dominant eigenvalue
        self.assertAlmostEqual(l[-1], 1 + (M-1) * rho, places=tolerance)

        # verify elements of the eigenvector corresponding to the 
        # dominant eigenvalue
        for i in range(M):
            self.assertAlmostEqual(v[i, -1], 1/np.sqrt(M), places=tolerance)

        # verify all eigenvalues other than dominant eigenvalue
        for li in l[:-1]:
            self.assertAlmostEqual(li, 1-rho, places=tolerance)

    def test_exceptions(self):
        """Verify that exceptions are raised.

        Test 1: the number of methods is an invalid integer
        Test 2: the value rho is outside designated interval
        """
        M, rho = 10, 0.2
        
        # Test 1
        with self.assertRaises(ValueError):
            sample._construct_corr_matrix(-1, rho)
        with self.assertRaises(TypeError):
            sample._construct_corr_matrix([3,4], rho)
        with self.assertRaises(TypeError):
            sample._construct_corr_matrix(4.3, rho)
        with self.assertRaises(TypeError):
            sample._construct_corr_matrix(4., rho)

        # Test 2
        with self.assertRaises(ValueError):
            sample._construct_corr_matrix(M, -0.3)
        with self.assertRaises(ValueError):
            sample._construct_corr_matrix(M, -2)
        with self.assertRaises(ValueError):
            sample._construct_corr_matrix(M, 1)
        with self.assertRaises(ValueError):
            sample._construct_corr_matrix(M, 54332)
        with self.assertRaises(ValueError):
            sample._construct_corr_matrix(M, 1.00)


class TestMultivariateGauss(TestCase):
    def test_exceptions(self):
        """Verify that exceptions are raised

        Test 1: Incorrect means
        Test 2: Incorrect covariance matrix
        Test 3: Incorrect number of samples
        """
        M = 4
        rho = 0.2
        N = 250
        m = np.zeros(M)
        c = sample._construct_corr_matrix(M, rho)

        # Test 1
        with self.assertRaises(AttributeError):
            sample.multivariate_gauss(m.tolist(), c, N)
        with self.assertRaises(AttributeError):
            sample.multivariate_gauss(1, np.array([[1]]), N)
        with self.assertRaises(ValueError):
            sample.multivariate_gauss(np.vstack([m, m]), c, N)


        # Test 2 
        with self.assertRaises(ValueError):
            sample.multivariate_gauss(m, c[:, :2], N)
        with self.assertRaises(ValueError):
            sample.multivariate_gauss(m, c[:2, :][:,:2], N)
        with self.assertRaises(ValueError):
            cp = np.triu(c)
            sample.multivariate_gauss(m, cp, N)
        with self.assertRaises(AttributeError):
            sample.multivariate_gauss(m, c.tolist(), N)
        with self.assertRaises(ValueError):
            cp = c.copy()
            cp[0,1] = 1-rho
            sample.multivariate_gauss(m, cp, N)

        # Test 3
        with self.assertRaises(ValueError):
            sample.multivariate_gauss(m, c, -1)
        with self.assertRaises(ValueError):
            sample.multivariate_gauss(m, c, 0)
        with self.assertRaises(TypeError):
            sample.multivariate_gauss(m, c, 5.5)

    def test_int_seed(self):
        """Verify that seed specification.

        Specifying a seed should result in reproducible samples from
        the multivariate Gaussian.

        Test 1: Given an array of means (m) and a covariance matrix (c)
            generate samples x, y specifying seed.  Each element of 
            x and y should be equal.
        Test 2: Given z, in which a seed is not specified, no element of
            x should be equal to that of z.
        """
        M = 4
        rho = 0.2
        N = 250
        seed = 15434
        m = np.zeros(M)
        c = sample._construct_corr_matrix(M, rho)

        x = sample.multivariate_gauss(m, c, N, seed=seed)
        y = sample.multivariate_gauss(m, c, N, seed=seed)
        z = sample.multivariate_gauss(m, c, N)

        # Test 1
        self.assertTrue((x == y).all())

        # Test 2
        self.assertFalse((x == z).any())

    def test_rng_seed(self):
        """Verify seed properties given np.random.default_rng() generator object

        Test 1: samples created from two new generators with same integer
            seed produce identical samples
        Test 2: generating N samples twice from a single 
            np.random.default_rng(seed) object instance is identical to
            sampling 2N samples once from a different instance of the same
            object
        Test 3: different seeds result in unique samples
        """
        M, N, rho = 10, 1000, 0.43
        seed = 12335
        m = np.zeros(M)
        c = sample._construct_corr_matrix(M, rho)

        # Test 1
        rngs = [np.random.default_rng(seed) for _ in range(10)]
        # Need to iterate over odds so that same generator isn't called twice
        for i in range(1, 10):
            # strategy for constraining i to odd values
            if i % 2 == 0:
                continue
            x = sample.multivariate_gauss(m, c, N, seed=rngs[i-1])
            y = sample.multivariate_gauss(m, c, N, seed=rngs[i])
            self.assertTrue((x == y).all())
                    
        # Test 2
        c = np.eye(M)
        rngs = [np.random.default_rng(seed) for _ in range(2)]
        x = sample.multivariate_gauss(m, c, N + N, seed=rngs[0])
        y = sample.multivariate_gauss(m, c, N, seed=rngs[1])
        z = sample.multivariate_gauss(m, c, N, seed=rngs[1])

        self.assertTrue((y != z).all())
        self.assertTrue((x[0,:N] == y[0,:]).all())
        x = x.flatten()
        w = np.hstack([y.flatten(), z.flatten()])
        self.assertTrue((x == w).all())

        # Test 3
        rngs = [np.random.default_rng(), 
                np.random.default_rng(seed),
                np.random.default_rng(seed-10)]

        r = list(range(len(rngs)))

        for i, rng in enumerate(rngs):
            r[i] = sample.multivariate_gauss(m, c, N, seed=rng)
        self.assertTrue((r[0] != r[1]).all())
        self.assertTrue((r[0] != r[2]).all())
        self.assertTrue((r[1] != r[2]).all())

class TestAUC(TestCase):
    """Test function _auc_2_delta."""
    def test_exceptions(self):
        """Verify inputs to _auc_2_delta function

        Test 1: auc out of interval, i.e <0 and > 0
        Test 2: v is not a tuple or a container  object of length 2
        Test 3: elements of v are negative
        """
        auc = 0.2
        v = (1, 1)

        # Test 1
        with self.assertRaises(ValueError):
            sample._auc_2_delta(-0.2, v)
        with self.assertRaises(ValueError):
            sample._auc_2_delta(1.2, v)

        # Test 2
        with self.assertRaises(TypeError):
            sample._auc_2_delta(auc, 1)
        with self.assertRaises(ValueError):
            sample._auc_2_delta(auc, (1,1,1))
        with self.assertRaises(TypeError):
            sample._auc_2_delta(auc, (1))

        # Test 3
        with self.assertRaises(ValueError):
            sample._auc_2_delta(auc, (-1, 1))
        with self.assertRaises(ValueError):
            sample._auc_2_delta(auc, (1, -1))

    def test_value(self):
        """Verfiy correct value returned.

        Test 1: auc=0.5 => delta = 0
        Test 2: auc < 0.5 => delta < 0
        Test 3: auc > 0.5 => delta > 0
        """
        auc_vals = np.linspace(0.1, 0.9, 9)
        v = (1, 1)

        for auc in auc_vals:
            if auc == 0.5:  # Test 1
                expr = sample._auc_2_delta(auc, v) == 0
            elif auc < 0.5: # Test 2
                expr = sample._auc_2_delta(auc, v) < 0
            else:           # Test 3
                expr = sample._auc_2_delta(auc, v) > 0

            self.assertTrue(expr)


class TestDataSet(TestCase):
    """Input validation / erroneous inputs"""
    def test_inputs(self):
        """Verify that invalid inputs raise exceptions.

        Test 1: auc value outside [0, 1]
        Test 2: corr_coef outside [0, 1)
        Test 3: prevalence outside (0,1) or a type other than float
        Test 4: N samples > 1, and can't be float
        """
        M, N, prevalence, corr_coef = 5, 1000, 0.3, 0.1
        auc = np.linspace(0.6, 0.9, M)
        
        # Test 1
        with self.assertRaises(ValueError):
            sample.data_set(1.2, corr_coef, prevalence, N)
        with self.assertRaises(ValueError):
            sample.data_set(-0.2, corr_coef, prevalence, N)
        with self.assertRaises(ValueError):
            sample.data_set(np.linspace(0, 1.1, M),  corr_coef, prevalence, N)
        with self.assertRaises(ValueError):
            sample.data_set(np.linspace(-0.23, 0.8, M),  corr_coef, prevalence, N)

        # Test 2
        with self.assertRaises(ValueError):
            sample.data_set(auc, 1.2, prevalence, N)
        with self.assertRaises(ValueError):
            sample.data_set(auc, 1, prevalence, N)
        with self.assertRaises(ValueError):
            sample.data_set(auc, -0.2, prevalence, N)

        # Test 3
        with self.assertRaises(ValueError):
            sample.data_set(auc, corr_coef, 0, N)
        with self.assertRaises(ValueError):
            sample.data_set(auc, corr_coef, 1, N)
        with self.assertRaises(TypeError):
            sample.data_set(auc, corr_coef, (0.5, 0.5), N)
        with self.assertRaises(TypeError):
            sample.data_set(auc, corr_coef, [0.5, 0.5], N)
        with self.assertRaises(ValueError):
            sample.data_set(auc, corr_coef, np.array([0.5, 0.5]), N)
        with self.assertRaises(TypeError):
            sample.data_set(auc, corr_coef, set(0.4, 0.5), N)
        with self.assertRaises(TypeError):
            sample.data_set(auc, corr_coef, set(0.4, 0.4), N)

        # Test 4
        with self.assertRaises(ValueError):
            sample.data_set(auc, corr_coef, 0.4, 1)
        with self.assertRaises(ValueError):
            sample.data_set(auc, corr_coef, 0.4, -41)
        with self.assertRaises(TypeError):
            sample.data_set(auc, corr_coef, 0.4, 34.2)


    def test_1d(self):
        """Validate 1-d sampling.

        Test 1: R, y dimensions are as expected
        Test 2: R is a rank array
        Test 3: y is an array of binary labels
        Test 4: positive class prevalence is recovered
        """
        N = 1000
        prevalence = 0.3
        RTRUE = np.arange(1, N+1)
        YTRUE = [0,1]

        R, y = sample.data_set(0.7, 0, prevalence, N)

        # Test 1
        self.assertEqual(y.ndim, 1)
        self.assertEqual(R.ndim, 2)
        self.assertEqual(R.shape, (1, N))
        self.assertEqual(y.size, N)

        # Test 2
        self.assertEqual(np.setdiff1d(RTRUE, R[0,:]).size, 0)
        self.assertEqual(np.setdiff1d(R[0, :], RTRUE).size, 0)

        # Test 3
        self.assertEqual(np.setdiff1d(y, YTRUE).size, 0)
        self.assertEqual(np.setdiff1d(YTRUE, y).size, 0)

        # Test 4
        self.assertEqual(np.sum(y), int(N*prevalence))
        

    def test_output(self):
        """Validate that output satisfies expected generic properties.

        Test 1: R, y dimensions are as expected
        Test 2: y is an ((N,) ndarray) of binary values
        Test 3: prevalence is recovered
        Test 4: R is an ((M, N) ndarray) of rank values
        """
        N_samples = np.array([100, 1000])
        prevalence = np.linspace(0.1, 0.9, 9)
        M_methods = np.arange(5, 10)
        corr_coeffs = np.linspace(0, 0.9, 10)

        # used to validate binary label output by set comparison
        YTRUE = [0,1]

        for m in M_methods:

            auc = np.linspace(0.5, 0.9, m)

            for n in N_samples:

                # used to validate rank output by set comparison
                RTRUE = np.arange(1, n+1)

                for prev in prevalence:

                    for corr_coef in corr_coeffs:

                        R, y = sample.data_set(auc, corr_coef, prev, n)

                        # Test 1
                        self.assertEqual(R.ndim, 2)
                        self.assertEqual(y.ndim, 1)
                        self.assertEqual(R.shape, (m, n))
                        self.assertEqual(R.shape[1], y.size)

                        # Test 2
                        self.assertEqual(np.setdiff1d(y, YTRUE).size, 0)
                        self.assertEqual(np.setdiff1d(YTRUE, y).size, 0)

                        # Test 3
                        self.assertEqual(np.sum(y), int(n * prev))

                        # Test 4
                        for i in range(m):
                            self.assertEqual(np.setdiff1d(R[i, :], RTRUE).size, 0)
                            self.assertEqual(np.setdiff1d(RTRUE, R[i, :]).size, 0)

    def test_seed(self):
        M, N, corr_coef, prev = 20, 1000, 0, 0.3
        auc= np.linspace(0.5, 0.9, M)
        seed = 341324

        R0, y0 = sample.data_set(auc, corr_coef, prev, N)
        R1, y1 = sample.data_set(auc, corr_coef, prev, N, seed=seed)
        R2, y2 = sample.data_set(auc, corr_coef, prev, N, seed=seed)
        R3, y3 = sample.data_set(auc, corr_coef, prev, N, seed=seed - 10)

        self.assertTrue((R1 == R2).all())
        self.assertTrue((y1 == y2).all())

        self.assertTrue(np.sum(R0 != R1) > 0.5*N)
        self.assertTrue(np.sum(R0 != R3) > 0.5*N)
        self.assertTrue(np.sum(R1 != R3) > 0.5*N)
    

if __name__ == "__main__":
    main()
