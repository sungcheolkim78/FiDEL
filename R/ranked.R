# ranked.R - ranked probability based algorithms
#
# Sungcheol Kim @ IBM
# 2020/01/15
#
# version 1.0

library(data.table)
library(ggpubr)

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

#' Calculate AUC using rank
#'
#' Function to calculate AUC from score and ground truth label using rank sum formula
#'
#' @param scores A list of values from binary classifier
#' @param y A list of labels
#' @param class1 A name of class 1
#' @return the area under receiver operating curve
#' @examples
#' auc.rank(scores, y)
#' @export
auc.rank <- function(scores, y) {
  # score and label

  stopifnot(length(scores) == length(y))
  if (is.null(attr(y, 'class1'))) {
    labels <- levels(y)
    class1 <- labels[[1]]
  } else {
    class1 <- attr(y, 'class1')
  }

  # calculate class 1 and class 2
  N <- length(scores)
  N1 <- sum(y == class1)
  N2 <- N - N1
  mat <- data.table(scores=scores, y=y)
  mat$rank <- frankv(scores, order=-1)

  res <- abs(sum(mat$rank[y == class1])/N1 - sum(mat$rank[y != class1])/N2)/N + 0.5
  return(res)
}

# characteristics based on the rank threshold
build_curve <- function(scores, y) {
  stopifnot(length(scores) == length(y))
  if (is.null(attr(y, 'class1'))) {
    labels <- levels(y)
    class1 <- labels[[1]]
    N1 <- sum(y == class1)
  } else {
    class1 <- attr(y, 'class1')
    N1 <- floor(attr(y, 'N')*attr(y, 'rho'))
  }
  N <- length(y)

  tic(sprintf('... build_curve calculation (N=%i)', N))

  # calculate curve data
  df <- data.table(score=scores, y=y)
  df <- setorder(df, score)

  df$rank <- seq(1, N)
  df$tpr <- cumsum(df$y == class1)/N1
  df$fpr <- cumsum(df$y != class1)/(N-N1)
  df$bac <- 0.5*(df$tpr + 1 - df$fpr)
  df$prec <- cumsum(df$y == class1)/df$rank

  # calculate metric
  auc_bac <- 2*mean(df$bac) - 0.5
  l1 <- loess(bac ~ rank, df, span=0.1)
  th_bac <- which.max(l1$fitted)/N
  rm(l1)
  b <- get_fermi(auc_bac, N1/N)
  aupr <- 0.5*N1/N*(1 + N/(N1*N1)*sum(df$prec[1:(N-2)]*df$prec[2:(N-1)]))
  ci_info <- var_auc_fermi(auc_bac, N1/N, N=N, method="integral")

  # register metrics
  attr(df, 'auc_bac') <- auc_bac
  attr(df, 'auprc') <- aupr
  attr(df, "th_bac") <- th_bac
  attr(df, 'rstar') <- b[[3]]
  attr(df, 'beta') <- b[[1]]
  attr(df, 'mu') <- b[[2]]
  attr(df, 'N1') <- N1
  attr(df, 'var_auc') <- ci_info[['var_auc']]
  attr(df, 'Pxy') <- ci_info[['Pxy']]
  attr(df, 'Pxxy') <- ci_info[['Pxxy']]
  attr(df, 'Pxyy') <- ci_info[['Pxyy']]
  attr(df, 'ci1') <- auc_bac - 1.96*sqrt(ci_info[['var_auc']])
  attr(df, 'ci2') <- auc_bac + 1.96*sqrt(ci_info[['var_auc']])
  toc()

  return(df)
}

# calculate lambda1,2 from auc, rho
lambda.auc <- function(auc, N=100.0, rho=0.5, full=FALSE) {
  costFunc1 <- function(l, rho, rs, N) {
    r <- seq(1, N, by = 1.0)
    sum1 <- sum(1/(1+exp(l[2]*r - l[1])))
    sum2 <- sum(r/(N+N*exp(l[2]*r - l[1])))

    (N*rho - sum1)^2 + (rs/N - sum2)^2
  }

  costFunc2 <- function(l, rho, rs, N) {
    r <- seq(1, N, by = 1.0)
    sum1 <- sum(1/(1+exp(-l[2]*r + l[1])))
    sum2 <- sum(r/(N+N*exp(-l[2]*r + l[1])))
    rs_m <- N*(1-rho)*(N*(1-rho)+1)/2

    (N*(1-rho) - sum1)^2 + (0.5*(N+1) - rs/N - sum2)^2
  }

  # check dataframe
  check <- c('auc', 'N', 'rho') %in% names(auc)
  if (all(check)) {
    auroc <- auc$auc
    N <- auc$N
    rho <- auc$rho
  } else {
    auroc <- auc
  }
  N <- as.numeric(N)
  rs <- 0.5*N*rho*(N+1) - N*N*rho*(1-rho)*(auroc - 0.5)

  l <- lambda.appr(auroc, N=N, rho=rho)
  #print(l)
  initial <- c(l[['l1']], l[['l2']])

  temp <- optim(initial, costFunc1, rho=rho, rs=rs, N=N, control=list(maxit=8000, reltol=1e-12))
  if (temp$value > 1e-4) {
    temp <- optim(initial, costFunc2, rho=rho, rs=rs, N=N, control=list(maxit=8000, reltol=1e-12))
  }

  #print(temp)
  if (temp$value > 1e-5) message(sprintf("not converged - auc: %.4f, rho: %.4f", auroc, rho))
  l1 <- -temp$par[1]
  l2 <- temp$par[2]
  rs <- 1/l2 * log((1 - rho)/rho) - l1/l2

  if (full)
    return(data.table(l1 = l1, l2 = l2, rs = rs, l10 = initial[[1]], l20 = initial[[2]], auc=auroc, rho=rho))
  else
    return(c(l1 = l1, l2 = l2, rs = rs))
}

# calculate lambda1, lambda2 from auc, rho (version 2)
lambda.appr <- function(auc, N=100.0, rho=0.5) {
  N <- as.numeric(N)
  l1_low <- log(1/rho - 1) - 12*N*(auc-0.5)/(N*N-1)*((N+1+N*rho)*0.5 - N*rho*auc)
  l2_low <- 12*N*(auc-0.5)/(N*N-1)

  temp <- sqrt(rho*(1-rho)*(1-2*(auc-0.5)))
  l1_high <- -2*rho/(sqrt(3)*temp)
  l2_high <- 2/(sqrt(3)*N*temp)

  alpha <- 2*(auc - 0.5)
  l1 <- l1_high*alpha + l1_low*(1-alpha)
  l2 <- l2_high*alpha + l2_low*(1-alpha)
  rs <- 1/l2 * log((1 - rho)/rho) - l1/l2

  return(c(l1low=l1_low, l2low=l2_low, l1high=l1_high, l2high=l2_high, l1=l1, l2=l2, rs=rs))
}

#' calculate beta, mu using normalized by N
#' @param auc A value of AUC
#' @param rho A value of prevalence
#' @param N A number of samples, when N=1, the results are normalized by N
#' @param resol A value for converting integral into summation
#' @param method 'beta' for beta, mu 'lambda' for lambda1, lambda2
#' @return in case of 'beta' - beta, mu, rstar
#' in case of 'lambda' - lambda1, lambda2, rstar
get_fermi <- function(auc, rho, N=1, resol=0.0001, method='beta') {
  # cost function - normalized version
  # mu' = mu/N
  # beta' = beta*N
  # r*' = r*/N
  cost <- function(bm, auc, rho, resol) {
    rprime <- seq(0, 1, by=resol)
    sum1 <- sum(resol/(1+exp(bm[1]*(rprime-bm[2]))))
    sum2 <- sum(resol*rprime/(1+exp(bm[1]*(rprime-bm[2]))))

    (rho - sum1)^2 + (0.5*rho - rho*(1-rho)*(auc-0.5) - sum2)^2
  }

  # auc should be in (0.5, 1.0)
  stopifnot(auc > 0.5)
  stopifnot(auc <= 1.0)

  # calculate intital beta' and mu'
  l <- lambda.appr(auc, N=1000, rho=rho)

  # optimize cost function
  initial <- c(l[['l2']]*1000, -l[['l1']]/(1000*l[['l2']]))
  temp <- optim(initial, cost, auc=auc, rho=rho, resol=resol, control=list(maxit=8000, reltol=1e-12))

  rs <- 1/temp$par[1] * log((1 - rho)/rho) + temp$par[2]

  if (method == 'beta') { return (c(beta=temp$par[1]/N, mu=temp$par[2]*N, rs=rs*N)) }
  else { return (c(l1=-temp$par[1]*temp$par[2], l2=temp$par[1]/N, rs=rs*N)) }
}

# plot histogram of score distribution
plot.scores <- function(scores, y) {
  y.flag <- TRUE
  if (missing(y)) {
    y <- rep('class1', length(scores))
    attr(y, 'class1') <- 'class1'
    attr(y, 'class2') <- 'class2'
    y.flag <- FALSE
  } else {
    if ('label' == class(y)) {
      l <- labels(y)
      attr(y, 'class1') <- l[[1]]
      attr(y, 'class2') <- l[[2]]
    }
  }
  df <- data.table(score=scores, y=y)

  score1 <- scores[y == attr(y, 'class1')]
  score2 <- scores[y == attr(y, 'class2')]

  g <- ggplot(data=df, aes(x=score, color=y)) + geom_histogram(alpha=0.5, position="identity", bins=50) +
    theme_classic() +
    theme(legend.position="top") +
    annotate("text", label=sprintf("mu1: %.4f", mean(score1)), x=mean(score1), y=0, vjust=0) +
    annotate("text", label=sprintf("N1: %d", length(score1)), x=mean(score1)-4*sd(score1), y=100, vjust=0)
  if (y.flag) {
    g <- g +
      annotate("text", label=sprintf("mu2: %.4f", mean(score2)), x=mean(score2), y=0, vjust=0) +
      annotate("text", label=sprintf("N2: %d", length(score2)), x=mean(score2)+4*sd(score2), y=100, vjust=0)
  }

  ggsave('score.pdf', width=8, height=6, dpi=300)
  return (g)
}

# plot curves of binary classifier
plot.curves <- function(df, filename='temp.pdf', type=-1) {
  # apply to score/y structure and pcr structure
  auc_bac <- attr(df, 'auc_bac')
  th_bac <- attr(df, 'th_bac')
  N <- length(df$bac)
  N1 <- attr(df, 'N1')
  idx <- floor(N*th_bac)
  idx_rs <- floor(N*attr(df, 'rstar'))

  g1 <- ggplot(data=df, aes(x=fpr, y=tpr)) + geom_line(alpha=0.7, size=1.5) +
    xlab("FPR") + ylab("TPR") + theme_classic() +
    geom_abline(intercept = 0, slope = 1, linetype="dashed") +
    annotate("point", x=df$fpr[idx], y=df$tpr[idx], color = "red", shape=18, size=3) +
    annotate("text", label=sprintf("AUC : %.4f", auc_bac), x=1, y=0, hjust=1)

  g2 <- ggplot(data=df, aes(x=tpr, y=prec)) + geom_line(alpha=0.7, size=1.5) +
    xlab("TPR") + ylab("Prec") + ylim(c(0,1)) + theme_classic() +
    geom_hline(yintercept = N1/N, linetype="dashed") +
    annotate("point", x=df$tpr[idx], y=df$prec[idx], color = "red", shape=18, size=3) +
    annotate("text", label=sprintf("AUPRC: %.4f", attr(df, 'auprc')),
             x=1, y=0, hjust=1)

  g3 <- ggplot(data=df, aes(x=rank, y=bac)) + geom_line(alpha=0.7, size=1) +
    xlab("Rank") + ylab("Balanced Accuracy") + ylim(c(0, 1)) + theme_classic() +
    geom_vline(xintercept = df$rank[idx], linetype="dashed") +
    annotate("point", x=df$rank[idx], y=df$bac[idx], color = cbPalette[7], shape=18, size=3) +
    annotate("point", x=df$rank[idx_rs], y=df$bac[idx_rs], color = cbPalette[4], shape=8, size=3) +
    annotate("text", x=N, y=0, hjust=1, vjust=0,
             label=sprintf("N: %.0f\np: %.2f", N, N1/N)) +
    annotate("text", label=sprintf("thr.: %.4f", th_bac), x=df$rank[idx], y=df$bac[idx]+0.1) +
    annotate("text", label=sprintf("r*: %.4f", attr(df,'rstar')), x=df$rank[idx_rs], y=df$bac[idx_rs]-0.1)

  if (type == -1) {
    g <- ggarrange(g1, g2, g3, labels=c("A", "B", "C"), ncol=3  , nrow=1)
    ggsave(filename, width=12, height=4, dpi=300)
  } else {
    if (type == 1) g <- g1
    if (type == 2) g <- g2
    if (type == 3) g <- g3
  }
  return (g)
}

# calculate ensemble score using fermi-dirac statistics
ensemble.fermi <- function(predictions, y, alpha = 1.0, method='+', debug.flag = FALSE) {
  M <- ncol(predictions)
  N <- nrow(predictions)

  res <- matrix(0, nrow=N, ncol=M)
  fd <- matrix(0, nrow=M, ncol=4)
  auclist <- apply(predictions, 2, auc.rank, y)
  fd[, 2:4] <- t(sapply(auclist, get_fermi, attr(y, 'rho'), N=N))
  fd[, 1] <- t(auclist)

  for(m in 1:M) {
    res[ , m] <- frankv(predictions[, m])
    if (method == '+')
      res[ , m] <- -fd[m, 2]^alpha *(fd[m, 4] - res[, m])
    else
      res[ , m] <- 12*N*(fd[m, 1] - 0.5)/(N*N - 1) *((N+1.)/2.- res[, m])
  }

  if(debug.flag) {
<<<<<<< HEAD
=======
    #temp <- as.data.frame(res)
    #temp[['summa+']] <- rowMeans(res)
    #plot(temp)
    #print(cor(temp, method = "spearman"))
>>>>>>> 065a1c3bf858d11a9bb8ea9613fd4d98ebcdf449
    print(fd)
  }

  return(rowSums(res))
}

# find eigen vector of estimate \hat{R} of the rank-one matrix R
# leading eigen vector is close to the true one
# adapted from r-summa (https://github.com/learn-ensemble/R-SUMMA/blob/master/R/matrix_calculations.R)
find_eigen <- function(covMatrix, tol=1e-6, niter_max=10000) {
  eig <- eigen(covMatrix, symmetric=TRUE)
  eig_all <- vector()

  l <- 0
  iter <- 1
  while (abs(l - eig$values[1]) > tol & (iter < niter_max)) {
    l <- eig$values[1]
    r <- (covMatrix - diag(diag(covMatrix)) + diag(l[1]*eig$vectors[, 1]^2))
    eig <- eigen(r, symmetric=TRUE)
    eig_all <- c(eig_all, eig$values[1])
    iter <- iter + 1
  }
  return (list(eig=eig, eig_all=eig_all))
}
