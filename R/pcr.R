# pcr.R - probability of class y at given rank r
#
# Sung-Cheol Kim @ IBM
#
# version 1.1.0 - 2020/03/12

library(ggpubr)
library(pROC)
library(latex2exp)

#' create dataframe of the probability of class with rank
#'
#' @param scores list of score values
#' @param y list of class values
#' @param sample_size the number of items in one sampling selection
#' @param sample_n the number of realization
#' @param method the method for calculate probability
#' @return the data class
new_pcr <- function(scores, y, sample_size=100, sample_n=300, method='bootstrap') {
  if (sample_size == 0) {
    sample_size <- length(scores)
    sample_n <- 1
  }

  # validate inputs
  stopifnot(length(scores) == length(y))
  if (is.null(attr(y, 'rho')) || attr(y, 'rho') == 0) { y <- to_label(y, class1=class1) }

  # calculation on pcr
  if (method == 'bootstrap') {
    prob <- pcr_sample(scores, y, sample_size=sample_size, sample_n=sample_n)
  }

  pcrd <- build_curve_pcr(prob)
  info <- attr(pcrd, 'info')
  info <- c(info, sample_size=sample_size, sample_n=sample_n, N.data=length(y), method=method)

  structure(pcrd,
            info=info,
            class=c("pcr", "data.table", "data.frame"))
}

validate_pcr <- function(x) {
  # check binary classes

  x
}

#' create S4 object of the probability of class with rank (PCR)
#'
#' @param scores list of score values
#' @param y list of class values
#' @param sample_size the number of items in one sampling selection
#' @param sample_n the number of realization
#' @return S4 object
pcr <- function(scores, y, sample_size=100, sample_n=300) {
  validate_pcr(new_pcr(scores, y, sample_size=sample_size, sample_n=sample_n))
}

#' calculate pcr using bootstrap method
#'
#' @param scores list of score values
#' @param y list of class values
#' @param sample_size the number of items in one sampling selection
#' @param sample_n the number of realization
#' @return the probability of class at given rank
pcr_sample <- function(scores, y, sample_size=100, sample_n=300) {
  if (length(y) - 10 < sample_size) {
    sample_size <- length(y) - 10
    message(paste0('... set sample size as ', sample_size, ' (original size: ', length(y), ')'))
  }

  N1_new <- floor(attr(y, 'rho')*sample_size)
  N2_new <- sample_size - N1_new
  totalidx <- seq_along(scores)
  c1_idx <- totalidx[y == attr(y, 'class1')]
  c2_idx <- totalidx[y == attr(y, 'class2')]

  get_prob <- function(seed) {
    set.seed(seed)
    idx <- c(sample(c1_idx, N1_new), sample(c2_idx, N2_new))
    tmp <- as.double(y[idx] == attr(y, 'class1'))
    return (tmp[order(scores[idx], decreasing = FALSE)])
  }

  seed_list <- sample(100*sample_n, sample_n)
  mat <- sapply(seed_list, get_prob)

  prob <- rowMeans(mat)

  return (prob)
}

#' TODO
#'
#' @param scores list of score values
#' @param y list of class values
#' @param nfold the number of items in one sampling selection
#' @return the probability of class at given rank
pcr_nfold <- function(scores, y, nfold=10) {
  c1_idx <- totalidx[y == attr(y, 'class1')]
  c2_idx <- totalidx[y == attr(y, 'class2')]
  if (min(c(length(c1_idx), length(c2_idx))) < nfold) {
    message(paste0('min sample size ', length(c1_idx), ' < nfold ', nfold))
  }
}

#' create ROC curve based on the rank threshold
#'
#' @param prob list of the probability
#' @return data frame with info
build_curve_pcr <- function(prob) {
  N <- length(prob)
  N1 <- sum(prob)
  N2 <- N - N1
  rho <- N1/N

  tic(sprintf('... build_curve calculation (N=%i)', N))

  # calculate curve data
  df <- data.table(rank=seq_along(prob), prob=prob)

  df$tpr <- cumsum(prob)/N1
  df$fpr <- cumsum(1-prob)/N2
  df$bac <- 0.5*(df$tpr + 1 - df$fpr)
  df$prec <- cumsum(prob)/df$rank

  # calculate metric
  auc0 <- auc.pcr(prob)
  # smooth curves and calculate optimal threshold
  l1 <- loess(bac ~ rank, df, span=0.1)
  # calculate beta and mu based on AUC and prevalence
  b <- get_fermi(auc0, rho)
  ci_info <- var_auc_fermi(auc0, rho, N=N, method="integral")

  # register metrics
  info <- list(auc_rank=auc0,
               auc_bac=2*mean(df$bac) - 0.5,
               auprc=0.5*N1/N*(1 + N/(N1*N1)*sum(df$prec[1:(N-2)]*df$prec[2:(N-1)])),
               th_bac=which.max(l1$fitted)/N,
               rstar=b[[3]],
               beta=b[[1]],
               mu=b[[2]],
               rho=rho,
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

#' calculate AUC from probability of class at given rank
#'
#' @param prob list of the probability
#' @return AUC
auc.pcr <- function(prob) {
  # rank and prob
  rank <- seq_along(prob)
  N <- length(prob)
  N1 <- sum(prob)
  N2 <- N - N1

  return (abs(sum(rank * prob)/N1 - sum(rank * (1 - prob))/N2)/N + 0.5)
}

#' calculate AUPRC from probability of class at given rank
#'
#' @param prob list of the probability
#' @return AUPRC
auprc.pcr <- function(prob) {
  N <- length(prob)
  rho <- sum(prob)/N

  prec <- cumsum(prob)/seq_along(prob)
  return (0.5*rho*(1 + sum(prec[1:(N-2)] * prec[2:(N-1)])/(N*rho*rho)))
}

sigma.rank <- function(rankprob, debug.flag = FALSE) {
  check <- c("rank", "prob") %in% names(rankprob)
  stopifnot(all(check))

  N1 <- sum(rankprob$prob)
  N2 <- length(rankprob$prob) - N1
  rmean <- sum(rankprob$rank * rankprob$prob)/N1
  r2mean <- sum(rankprob$rank * rankprob$rank * rankprob$prob)/N1

  if (debug.flag) {
    print(paste0('N: ', length(rankprob$prob)))
    print(paste0('N1: ', sum(rankprob$prob)))
    print(paste0('variance <r|1>: ', r2mean - rmean*rmean))
    print(paste0('sigma <r|1>: ', sqrt(r2mean - rmean*rmean)))
    print(paste0('sigma AUC: ', sqrt(r2mean - rmean*rmean)/N2))
  }

  return (sqrt(r2mean - rmean*rmean))
}

#' compare pcr with fermi-dirac distribution
#'
#' @param pcrd pcr dataframe
#' @return cor, p.value, MAE, RMS, SSEV
check.pcr <- function(pcrd) {
  info <- attr(pcrd, 'info')
  fy <- fermi.b(pcrd$rank, info$beta, info$mu, normalized = TRUE)
  err <- pcrd$prob - fy

  co <- cor.test(pcrd$prob, fy)
  MAE <- mean(abs(err))
  RMSE <- sqrt(mean(err*err))
  SSEV <- sum(err*err)/var(err)

  return (c(cor=co$estimate, p.value=co$p.value, MAE=MAE, RMSE=RMSE, SSEV=SSEV))
}

#' plot pcr with fermi-dirac distribution
#'
#' @param pcrd pcr dataframe
#' @return ggplot
plot.pcr <- function(pcrd, fname='') {
  df <- data.table(x=pcrd$rank, y=pcrd$prob)
  info <- attr(pcrd, 'info')

  fy <- fermi.b(pcrd$rank, info$beta/info$sample_size, info$mu*info$sample_size)
  co <- cor.test(pcrd$prob, fy)
  print(co)

  msg2 <- sprintf("Beta: %.3g\nMu: %.3g", info$beta/info$sample_size, info$mu*info$sample_size)
  msg3 <- sprintf("N.data: %d\nsampling #: %d\nAUC: %.4f\nprevalence: %.3f",
                  info$N.data, info$sample_n, info$auc_rank, info$rho)
  idx <- floor(info$sample_size/2)

  g <- ggplot(data=df) + geom_point(aes(x=x, y=y)) +
    geom_line(aes(x=x, y=fy), linetype="dashed", color="green") +
    #geom_errorbar(aes(x=x, ymin=y-sd, ymax=y+sd), alpha=0.5, position=position_dodge(0.05)) +
    theme_classic() + ylab('P(1|r)') + xlab('Rank') +
    #annotate("text", label=msg1, x=0, y=0, hjust=0, vjust=0) +
    annotate("text", label=msg2, x=idx, y=fy[idx], vjust=0, hjust=0) +
    annotate("text", label=msg3, x=max(pcrd$rank), y=1.0, hjust=1, vjust=1) +
    ggtitle(paste0('Pearson Correlation r=', round(co$estimate, digits = 3),
                   ' p=', format(co$p.value, nsmall=3), ' between PCR and FD\n',
                   'method: ', info$method))

  if (fname == '') {
    fname <- paste0("N", info$sample_size, "M", info$sample_n, ".pdf")
  }
  ggsave(fname, width=7, height=4)
  return (g)
}
