
from unittest import TestCase, main

import numpy as np
from fd import sample
from fd import stats

class TestDelta(TestCase):
    """Test data validation and delta for the perfect classifier."""
    def setUp(self):
        self.N = 100
        self.N1 = 30
        self.r = np.arange(self.N) + 1
        self.y = np.zeros(self.N)
        self.y[:self.N1] = 1

    def test_2d_rank(self):
        """Verify exception raised for 2-d rank inputs"""
        with self.assertRaises(ValueError):
            self.r = np.vstack([self.r, self.r])
            stats.delta(self.r, self.y)

    def test_2d_rank_and_labels(self):
        """Verify that labels and rank must be 1-d."""
        with self.assertRaises(ValueError):
            self.r = np.vstack([self.r, self.r])
            self.y = np.vstack([self.y, self.y])
            stats.delta(self.r, self.y)

    def test_rank_label_shape_requirement(self):
        """Verify that the shape of rank and label arrays match."""
        with self.assertRaises(ValueError):
            stats.delta(self.r[:-1], self.y)

    def test_rank_values_requirement(self):
        """Verify that ranks require values [1,N], no ties."""
        with self.assertRaises(ValueError):
            self.r[10] = self.r[11]
            stats.delta(self.r, self.y)

    def test_label_values_requirement(self):
        """Verify class labels data are 0 or 1."""
        with self.assertRaises(ValueError):
            self.y[10] = -1
            stats.delta(self.r, self.y)

    def test_all_labels_zero(self):
        """Verify that all samples can't be class 0."""
        with self.assertRaises(ValueError):
            self.y = np.zeros(self.N)
            stats.delta(self.r, self.y)

    def test_all_labels_ones(self):
        """Verify taht all samples can't be class 1."""
        with self.assertRaises(ValueError):
            self.y = np.ones(self.N)
            stats.delta(self.r, self.y)

    def test_perfect_classifier(self):
        """Verify that delta = N/2 for the perfect classifier."""
        self.assertEqual(stats.delta(self.r, self.y), self.N/2)

    def test_opposing_convention(self):
        """Verify that delta = -N/2 for the worst classifier.

        The worst classifier is one which is perfect by adopts the opposite
        convention, that is positive class samples assigned high rank.
        """
        self.r = self.N+1 - self.r
        self.assertEqual(stats.delta(self.r, self.y), -self.N/2)


class TestRank2AUC(TestCase):
    """Test data AUC calculation for the perfect classifier."""
    def setUp(self):
        self.N = 100
        self.N1 = 30
        self.r = np.arange(self.N) + 1
        self.y = np.zeros(self.N)
        self.y[:self.N1] = 1

    def test_perfect_classifier(self):
        """Verify that AUC=1 for the perfect classifier."""
        self.assertEqual(stats.rank_2_auc(self.r, self.y), 1.)

    def test_opposing_convention(self):
        """Verify that AUC = 0 for the worst classifier.

        The worst classifier is one which is perfect by adopts the opposite
        convention, that is positive class samples assigned high rank.
        """
        self.r = self.N+1 - self.r
        self.assertEqual(stats.rank_2_auc(self.r, self.y), 0.)



if __name__ == "__main__":
    main()
