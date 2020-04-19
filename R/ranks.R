# ranks.R - rank based metric calculations for structured learning
#
# Sungcheol Kim @ IBM
#
# version 1.0.0 - 2020/01/15
# version 1.0.1 - 2020/04/13 - clean up

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
auc.rank <- function(scores, y, class1=NULL) {
  # validate inputs
  stopifnot(length(scores) == length(y))
  if (is.null(attr(y, 'rho')) || attr(y, 'rho') == 0) { y <- to_label(y, class1=class1) }

  # calculate class 1 and class 2
  N <- attr(y, 'N')
  N1 <- attr(y, 'N1')
  N2 <- attr(y, 'N2')
  mat <- data.table(scores=scores, y=y)
  mat$rank <- frankv(scores, order=-1)

  res <- abs(sum(mat$rank[y == attr(y, 'class1')])/N1 - sum(mat$rank[y == attr(y, 'class2')])/N2)/N + 0.5

  if (res < 0.5) {
    message('... class label might be wrong.')
    res <- 1 - 0.5
  }

  return (res)
}

# characteristics based on the rank threshold
build_curve <- function(scores, y, class1=NULL) {
  stopifnot(length(scores) == length(y))
  if (is.null(attr(y, 'rho')) || attr(y, 'rho') == 0) { y <- to_label(y, class1=class1) }

  N <- attr(y, 'N')
  N1 <- attr(y, 'N1')
  class1 <- attr(y, 'class1')

  tic(sprintf('... build_curve calculation (N=%i)', N))

  # calculate curve data
  df <- data.table(score=scores, y=y)
  df <- setorder(df, score)

  df$rank <- seq(1, N)
  df$tpr <- cumsum(df$y == attr(y, 'class1'))/attr(y, 'N1')
  df$fpr <- cumsum(df$y == attr(y, 'class2'))/attr(y, 'N2')
  df$bac <- 0.5*(df$tpr + 1 - df$fpr)
  df$prec <- cumsum(df$y == attr(y, 'class1'))/df$rank

  # calculate metric
  auc0 <- auc.rank(scores, y, class1=class1)
  # smooth curves and calculate optimal threshold
  l1 <- loess(bac ~ rank, df, span=0.1)
  # calculate beta and mu based on AUC and prevalence
  b <- get_fermi(auc0, attr(y, 'rho'))
  ci_info <- var_auc_fermi(auc0, attr(y, 'rho'), N=N, method="integral")

  # register metrics
  info <- list(auc_rank=auc0,
               auc_bac=2*mean(df$bac) - 0.5,
               auprc=0.5*N1/N*(1 + N/(N1*N1)*sum(df$prec[1:(N-2)]*df$prec[2:(N-1)])),
               th_bac=which.max(l1$fitted)/N,
               rstar=b[[3]],
               beta=b[[1]],
               mu=b[[2]],
               rho=attr(y, 'rho'),
               var_auc=ci_info[['var_auc']],
               Pxy=ci_info[['Pxy']],
               Pxxy=ci_info[['Pxxy']],
               Pxyy=ci_info[['Pxyy']],
               ci1=auc0 - 1.96*sqrt(ci_info[['var_auc']]),
               ci2=auc0 + 1.96*sqrt(ci_info[['var_auc']])
  )
  rm(l1, b, ci_info)

  attr(df, 'info') <- info
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
  stopifnot(auc <= 1.0)
  if (auc < 0.5) { auc <- 1.0 - auc }

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
plot.scores <- function(scores, y, class1=NULL) {
  y.flag <- TRUE
  if (missing(y)) {
    if (is.null(class1)) { class1 <- 'Class1'}
    y <- rep(class1, length(scores))
    y <- to_label(y, class1=class1)
    y.flag <- FALSE
  } else {
    if (is.null(attr(y, 'rho')) || attr(y, 'rho') == 0) { y <- to_label(y, class1=class1) }
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
  info <- attr(df, 'info')
  N <- length(df$bac)
  idx <- floor(N*info$th_bac)
  idx_rs <- floor(N*info$rstar)

  g1 <- ggplot(data=df, aes(x=fpr, y=tpr)) + geom_line(alpha=0.7, size=1.5) +
    xlab("FPR") + ylab("TPR") + theme_classic() +
    geom_abline(intercept = 0, slope = 1, linetype="dashed") +
    annotate("point", x=df$fpr[idx], y=df$tpr[idx], color = "red", shape=18, size=3) +
    annotate("text", x=1, y=0, hjust=1,
             label=sprintf("AUC(BAC) : %.4f\nAUC(Rank) : %.4f", info$auc_bac, info$auc_rank))

  g2 <- ggplot(data=df, aes(x=tpr, y=prec)) + geom_line(alpha=0.7, size=1.5) +
    xlab("TPR") + ylab("Prec") + ylim(c(0,1)) + theme_classic() +
    geom_hline(yintercept = info$rho, linetype="dashed") +
    annotate("point", x=df$tpr[idx], y=df$prec[idx], color = "red", shape=18, size=3) +
    annotate("text", label=sprintf("AUPRC: %.4f", info$auprc),
             x=1, y=0, hjust=1)

  g3 <- ggplot(data=df, aes(x=rank, y=bac)) + geom_line(alpha=0.7, size=1) +
    xlab("Rank") + ylab("Balanced Accuracy") + ylim(c(0, 1)) + theme_classic() +
    geom_vline(xintercept = df$rank[idx], linetype="dashed") +
    annotate("point", x=df$rank[idx], y=df$bac[idx], color = cbPalette[7], shape=18, size=3) +
    annotate("point", x=df$rank[idx_rs], y=df$bac[idx_rs], color = cbPalette[4], shape=8, size=3) +
    annotate("text", x=N, y=0, hjust=1, vjust=0,
             label=sprintf("N: %.0f\np: %.2f", N, info$rho)) +
    annotate("text", label=sprintf("thr.: %.4f", info$th_bac), x=df$rank[idx], y=df$bac[idx]+0.1) +
    annotate("text", label=sprintf("r*: %.4f", info$rstar), x=df$rank[idx_rs], y=df$bac[idx_rs]-0.1)

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
