# Fermi-Dirac based Ensemble Learning (FiDEL)

an ensemble learning algorithm that uses the calibrated nature of the output probability of possibly very different classifiers in an unsupervised setting well-suited for problems when we have scant data at hand to apply supervised ensemble learning.

## Installation

required packages: kernlab, lme4, coin, modeltools, arm, party, caret, earth, RSNNS, C50, glmnet

Use R-Studio and load `FiDEL.Rproj` and build/install

## Examples

### Figure 1 in PNAS

```{r}
library(FiDEL)
# generate labels
y <- create.labels(N = 100000, rho=0.5)

# generate scores with a specific AUC from gaussian distribution
gs <- create.scores.gaussian(y, auc=0.9, tol = 0.0001)

# create pcr data
pcrd <- pcr(gs, y, sample_size=100, sample_n=1000)

# save plot
plot.pcr(pcrd, fname='results/Figure1.pdf')
```

### Figure 2 in PNAS

### Figure 3 in PNAS

### Figure 4 in PNAS

