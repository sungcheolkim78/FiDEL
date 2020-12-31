# ci.R - Confidence Interval
#
# Sung-Cheol Kim @ IBM
# 2020/12/28
#
# version 1.0
# version 1.1 - using normalized beta, mu
# version 1.2 - add comments

library(tictoc)

# Sampling Method

#' count cases of Pxy
#'
#' @param pcr_m PCR data frame
#' @param seed A number for random seed
#' @return 1 or 0
countXY <- function(pcr_m, seed=1) {
  set.seed(seed)
  N <- length(pcr_m$prob)
  dices <- runif(N)
  check <- (pcr_m$prob > dices)             # make class 1 or 2 from class probability

  r1 <- sample(pcr_m$rank[check], 1)        # choose a rank from class 1
  r2 <- sample(pcr_m$rank[!check], 1)       # choose a rank from class 2

  # check conditions for P_xy
  if (r1 < r2) {    # rank of class 1 is less than that of class 2
    return (1)
  } else {
    return (0)
  }
}

#' count cases of Pxxy
#'
#' @param pcr_m PCR data frame
#' @param seed A number for random seed
#' @return 1 or 0
countXXY <- function(pcr_m, seed=1) {
  set.seed(seed)
  N <- length(pcr_m$prob)
  dices <- runif(N)
  check <- (pcr_m$prob > dices)             # make class 1 or 2 from class probability

  r1 <- sample(pcr_m$rank[check], 2)        # choose a rank from class 1
  r2 <- sample(pcr_m$rank[!check], 1)       # choose a rank from class 2

  # check conditions for P_xxy
  if (max(r1) < r2) {                   # rank of class 2 is the highest rank
    return(1)
  } else {
    return(0)
  }
}

#' count cases of Pxyy
#'
#' @param pcr_m PCR data frame
#' @param seed A number for random seed
#' @return 1 or 0
countXYY <- function(pcr_m, seed=1) {
  set.seed(seed)
  N <- length(pcr_m$prob)
  dices <- runif(N)
  check <- (pcr_m$prob > dices)             # make class 1 or 2 from class probability

  r1 <- sample(pcr_m$rank[check], 1)        # choose a rank from class 1
  r2 <- sample(pcr_m$rank[!check], 2)       # choose a rank from class 2

  # check conditions for P_xyy
  if (r1 < min(r2)) {                  # rank of class 1 is the lowest rank
    return (1)
  } else {
    return (0)
  }
}

#' calculate Pxy with sampling
#'
#' @param pcr_m PCR data frame
#' @param iter A number of iteration
#' @param debug.flag boolean value to show plot
#' @return Pxy (=AUC)
Pxy.sample <- function(pcr_m, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXY(pcr_m, seed=.x))
  res <- cumsum(res)/(1:iter)
  #print(res[length(res)])
  if (debug.flag) plot(res)

  return (res[length(res)])
}

#' calculate Pxxy with sampling
#'
#' @param pcr_m PCR data frame
#' @param iter A number of iteration
#' @param debug.flag boolean value to show plot
#' @return Pxxy
Pxxy.sample <- function(pcr_m, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXXY(pcr_m, seed=.x))
  res <- cumsum(res)/(1:iter)

  if (debug.flag) plot(res)

  return (res[length(res)])
}

#' calculate Pxyy with sampling
#'
#' @param pcr_m PCR data frame
#' @param iter A number of iteration
#' @param debug.flag boolean value to show plot
#' @return Pxyy
Pxyy.sample <- function(pcr_m, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXYY(pcr_m, seed=.x))
  res <- cumsum(res)/(1:iter)

  if (debug.flag) plot(res)

  return (res[length(res)])
}

# Integral method

#' calculate Pxy with integral formula
#'
#' @param beta A number for classifier temperature (beta)
#' @param mu A number for classifier chemical potential (mu)
#' @param rho the prevalence of dataset
#' @param resol the resolution of integral summation
#' @return Pxy
Pxy_int <- function(beta, mu, rho, resol=1e-6) {
  tic(sprintf('... Pxy integral calculation with res = %.3g', resol))
  ptable <- fermi.b(seq(0,1,by=resol), beta, mu)

  A <- ptable
  B <- rev(seq_along(ptable)) - rev(cumsum(rev(ptable)))
  res <- sum(A * B) - sum(ptable * (1-ptable))
  res <- res * resol * resol / (rho*(1-rho))
  toc()

  return(res)
}

#' calculate Pxxy with integral formula
#'
#' @param beta A number for classifier temperature (beta)
#' @param mu A number for classifier chemical potential (mu)
#' @param rho the prevalence of dataset
#' @param resol the resolution of integral summation
#' @return Pxxy
Pxxy_int <- function(beta, mu, rho, resol=1e-6) {
  tic(sprintf('... Pxxy integral calculation with res = %.3g', resol))
  ptable <- fermi.b(seq(0,1,by=resol), beta, mu)

  A <- 2*ptable*cumsum(ptable) - ptable^2
  B <- rev(seq_along(ptable)) - rev(cumsum(rev(ptable)))
  res <- sum(A * B) - sum(ptable^2 * (1-ptable)) - sum(ptable^2) - 2*sum(ptable*(1-ptable))
  res <- res * resol^3 / (rho^2 * (1-rho))
  toc()

  return(res)
}

#' calculate Pxyy with integral formula
#'
#' @param beta A number for classifier temperature (beta)
#' @param mu A number for classifier chemical potential (mu)
#' @param rho the prevalence of dataset
#' @param resol the resolution of integral summation
#' @return Pxyy
Pxyy_int <- function(beta, mu, rho, resol=1e-6) {
  tic(sprintf('... Pxyy integral calculation with res = %.3g', resol))
  ptable <- fermi.b(seq(0,1,by=resol), beta, mu)

  A <- ptable
  B <- rev(cumsum(rev(1-ptable)))^2
  res <- sum(A * B) - sum(ptable * (1-ptable)^2) - sum((1-ptable)^2) - 2*sum(ptable*(1-ptable))
  res <- res * resol^3 / (rho * (1-rho)^2)
  toc()

  return(res)
}

# Confidence interval calculation

#' calculate variance of AUC
#'
#' @param pcr_m PCR data frame
#' @param debug.flag boolean value for debug option
#' @return variance of AUC
var_auc_pcr <- function(pcr_m, debug.flag=FALSE) {
  N <- length(pcr_m$prob)
  N1 <- sum(pcr_m$prob)
  N2 <- N - N1

  Pxxy_value <- Pxxy.sample(pcr_m)
  Pxyy_value <- Pxyy.sample(pcr_m)
  Pxy_value <- Pxy.sample(pcr_m)

  var_auc <- (Pxy_value*(1 - Pxy_value) +
                (N1 - 1)*(Pxxy_value - Pxy_value*Pxy_value) +
                (N2 - 1)*(Pxyy_value - Pxy_value*Pxy_value))/(N1*N2)

  if (debug.flag) {
    print(paste0('Method: ', method))
    print(paste0('Rho: ', N1/N))
    print(paste0('Pxxy: ', Pxxy_value))
    print(paste0('Pxyy: ', Pxyy_value))
    print(paste0('Pxy (AUC): ', Pxy_value))
    print(paste0('AUC Sigma: ', sqrt(var_auc)))
    print(paste0('95% CI: ', Pxy_value - 1.96*sqrt(var_auc), ' - ', Pxy_value + 1.96*sqrt(var_auc)))
  }

  return(var_auc)
}

#' calculate variance of AUC from Fermi-Dirac distribution
#'
#' @param auc a value for AUC
#' @param rho the prevalence of the data set
#' @param N the size of the data set
#' @param iter the iteration number
#' @param method the Pxy, Pxxy, Pxyy calculation method - 'sampling', 'integral'
#' @param debug.flag boolean value for debug option
#' @return var_auc: variance of AUC, Pxy: Pxy, Pxxy: Pxxy, Pxyy: Pxyy
var_auc_fermi <- function(auc, rho, N=1000, iter=5000, method='sampling', debug.flag=FALSE) {
  # get FD variables
  b <- get_fermi(auc, rho, N=N)
  N1 <- floor(N*rho)
  N2 <- N - N1

  if (method == 'sampling') {
    ranks <- seq(1, N)
    probs <- fermi.b(ranks, b[[1]], b[[2]])
    #if (debug.flag) plot(ranks, probs)
    cm <- data.frame(rank=ranks, prob=probs)

    Pxxy_value <- Pxxy.sample(cm, iter=iter)
    Pxyy_value <- Pxyy.sample(cm, iter=iter)
    Pxy_value <- Pxy.sample(cm, iter=iter)
  } else {
    Pxxy_value <- Pxxy_int(b[[1]]*N, b[[2]]/N, rho)
    Pxyy_value <- Pxyy_int(b[[1]]*N, b[[2]]/N, rho)
    Pxy_value <- Pxy_int(b[[1]]*N, b[[2]]/N, rho)
  }

  var_auc <- (Pxy_value*(1 - Pxy_value) +
                (N1 - 1)*(Pxxy_value - Pxy_value*Pxy_value) +
                (N2 - 1)*(Pxyy_value - Pxy_value*Pxy_value))/(N1*N2)

  if (debug.flag) {
    print(paste0("Method: ", method))
    print(paste0("beta': ", b[[1]]*N))
    print(paste0("mu': ", b[[2]]/N))
    print(paste0('Pxxy: ', Pxxy_value))
    print(paste0('Pxyy: ', Pxyy_value))
    print(paste0('Pxy: ', Pxy_value))
    print(paste0('AUC: ', auc))
    print(paste0('AUC Sigma: ', sqrt(var_auc)))
    print(paste0('95% CI: ', auc - 1.96*sqrt(var_auc), ' - ', auc + 1.96*sqrt(var_auc)))
  }

  return(c(var_auc=var_auc, Pxy=Pxy_value, Pxxy=Pxxy_value, Pxyy=Pxyy_value))
}
