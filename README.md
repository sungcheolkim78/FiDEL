# Fermi-Dirac based Ensemble Learning (FiDEL)

an ensemble learning algorithm that uses the calibrated nature of the output probability of possibly very different classifiers in an unsupervised setting well-suited for problems when we have scant data at hand to apply supervised ensemble learning.

## Installation

- required packages: caret, tidyverse, tictoc, doParallel, data.table, ggpubr
- recommended packages for various classifiers: kernlab, lme4, coin, modeltools, arm, party, earth, RSNNS, C50, glmnet

Clone the repository and then load `FiDEL.Rproj` using R-Studio and build/install

or

```
R> install.packages('devtools')
R> install_github('sungcheolkim78/FiDEL')
```

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

```{r}
library(FiDEL)

# create beta mu dataframe
auclist <- (2:48)*0.01 + 0.5
rholist <- (2:18)*0.05
res <- create_beta_mu(auclist, rholist, N=1)

# convert to matrix
rhoN <- length(unique(res$rho))
AUCN <- length(unique(res$AUC))

rho <- unique(res$rho)
AUC <- unique(res$AUC)

beta <- matrix(res$beta, nrow=rhoN, ncol=AUCN)
mu <- matrix(res$mu, nrow=rhoN, ncol=AUCN)

# plot
pdf('results/Figure2.pdf', width=12, height=6)
par(mfrow=c(1, 2))
persp(rho, AUC, beta, theta = 30, phi = 15, shade=.3, ticktype='detailed', expand=.8, scale=T)
persp(rho, AUC, mu, theta = 30, phi = 15, shade=.3, ticktype='detailed', expand=.8, scale=T)
dev.off()
```

### Figure 3 in PNAS

create dataframe with AUC list and the prevalence list. This might take around 3 minutes.

```{r}
rholist <- c(0.1, 0.3, 0.5, 0.7, 0.9)
auclist <- create.auclist(0.6, 0.98, 10)
N0 <- 50000

res <- data.table()

for(r in rholist) {
  for (a in auclist) {
    # create test sets using AUC and prevalence
    y <- create.labels(N=N0, rho=r)
    gs <- create.scores.gaussian(y, auc=a)
    ds <- build_curve(gs, y)
    info <- attr(ds, 'info')
    
    # calculate confidence interval using pROC package
    ds_roc <- roc(y, gs)
    ds_ci <- ci(ds_roc)
    
    # data frame
    tmp <- data.table(N=length(y), rho=attr(y,'rho'), auc=a, auc_bac=info$auc_bac,
             auprc=info$auprc, th_bac=info$th_bac, rstar=info$rstar, 
             pxxy=info$Pxxy, pxyy=info$Pxyy, sig_auc=sqrt(info$var_auc), 
             sig_auc_delong=(ds_ci[2]-ds_ci[1])*.5)
    res <- rbind(res, tmp)
    rm(ds)
    rm(ds_roc)
  }
}
```

```{r}
# create sub-figure A
g1 <- ggplot(data=tmp) +
  geom_point(aes(y=rstar, x=th_bac)) + 
  geom_abline(slope=1, linetype='dashed') + theme_classic() +
  ylab(TeX('r_{FD}/N')) + xlab(TeX('r_{bac}/N')) +
  xlim(c(0.1,0.9)) + ylim(c(0.1,0.9))
g1
```

```{r}
# create sub-figure B
gb2 <- ggplot(data=res) + 
  geom_point(aes(x=sig_auc_delong, y=sig_auc)) + 
  geom_abline(slope=1, linetype='dashed') + theme_classic() + 
  xlab(TeX('$\\sigma_{AUC}$ (Delong)')) + ylab(TeX('$\\sigma_{AUC}$ (FD)')) + 
  theme(legend.position='none')
gb2
```

```{r}
library(ggpubr)
g <- ggarrange(g1, gb2, labels=c('A', 'B'), ncol=2, nrow=1)
ggsave("results/Figure3.pdf", width=7, height=3.5)
```

### Figure 4 in PNAS

