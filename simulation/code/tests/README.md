Fermi-Dirac Distribution Package Tests
======================================

The programs:

    - `inspect_gauss_sampling.py`
    - `inspect_rank_sampling.py`

print plots to pdf files for visual inspection of sampling methods.  The plots are stored in directories `gauss_sample_plots` or `rank_sample_plots`, respectively, unless specified otherwise.  If these directories don't exist they will be created in the current working directory.  These programs do not check whether any specific files exist, and repeated runs will result in overwriting previous plots.

I used the `unittest` module to run tests in the following files:

    - `test_classifiers.py`
    - `test_sampling.py`
    - `test_stats.py`
