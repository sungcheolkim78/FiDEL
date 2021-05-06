Fermi-Dirac Distriubtion for Binary Classification
==================================================

The package contains an implementation of the Fermi-Dirac distribution for ensemble learning described in:

```
Kim, S. et. al. "Learning from Fermions: the Fermi-Dirac Distribution Provides a Calibrated Probabilistic Output for Binary Classifiers" in review (2021)
```

Dependencies
------------

The package was developed and applied using:

- Python          3.7.3
- numpy           1.19.2
- scipy           1.4.1
- matplotlib      3.2.2

as well as any packages that `numpy`, `scipy`, and `matplotlib` require.


Example
-------

In this example I show how to use the `fd` package to simulate data, apply the
available classifiers, and compute their performance as measured by AUC.

First, import relevant packages

```Python
>>> import numpy as np
>>> from fd import classifiers as cls
>>> from fd import stats, sample
```

Then we need to set simulation parameters.  These include:

- `M` for the number of base classifiers,
- `N` for the total number of samples
- `prevalence` the fraction of samples belonging to the positive class
- `conditional_corr` the conditional correlation coefficient between base classifier predictions.

and lastly we need to make an array of AUC values, in which the ith element corresponds to the performance of the ith base classifier.

```Python
>>> M, N, prevalence, conditional_corr = 10, 1000, 0.3, 0
>>> auc = np.linspace(0.55, 0.85, M)
>>> R, y = sample.data_set(auc, conditional_corr, prevalence, N)
```

With these synthetic data, we can fit the FD model and find the best individual base classifier.

```Python
>>> fcl = cls.FDensemble()
>>> fcl.fit(R, y)

# the wisdom-of-crowd (Woc) classifier does not requires fitting.
>>> wcl = cls.Woc()

>>> bcl = cls.BestInd()
>>> bcl.fit(R, y)
```

Lastly, we compute the AUC of each classifier,

```Python
>>> classifiers = {"FDensemble": fcl, "Woc": wcl, "Best Ind": bcl}
>>> test_R, test_y = sample.data_set(auc, conditional_corr, prevalence, N)
>>> for key, cl in classifiers.items():
>>>     cl_auc = stats.rank_2_auc(cl.compute_ranks(test_R), test_y)
>>>     print("AUC of the {}: {}".format(key, cl_auc))
```

