# ci.R - Confidence Interval
#
# Sungcheol Kim @ IBM
# 2020/02/14
#
# version 1.0
# version 1.1 - using normalized beta, mu

# Sampling Method
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

auc.Pxysample <- function(pcr_m, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXY(pcr_m, seed=.x))
  res <- cumsum(res)/(1:iter)
  #print(res[length(res)])
  if (debug.flag) plot(res)

  return (res[length(res)])
}

Pxxy.sample <- function(pcr_m, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXXY(pcr_m, seed=.x))
  res <- cumsum(res)/(1:iter)

  if (debug.flag) plot(res)

  return (res[length(res)])
}

Pxyy.sample <- function(pcr_m, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXYY(pcr_m, seed=.x))
  res <- cumsum(res)/(1:iter)

  if (debug.flag) plot(res)

  return (res[length(res)])
}

# Integral method
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
var_auc_pcr <- function(pcr_m, method='Sampling', debug.flag=FALSE) {
  N <- length(pcr_m$prob)
  N1 <- sum(pcr_m$prob)
  N2 <- N - N1

  if (method == 'Sampling') {
    Pxxy_value <- Pxxy.sample(pcr_m)
    Pxyy_value <- Pxyy.sample(pcr_m)
    Pxy_value <- auc.Pxysample(pcr_m)
  } else if (method == 'Sum') {
    Pxxy_value <- Pxxy.sum2(pcr_m)
    Pxyy_value <- Pxyy.sum2(pcr_m)
    Pxy_value <- auc.Pxysum(pcr_m)
  }
  var_auc <- (Pxy_value*(1-Pxy_value) +
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
    Pxy_value <- auc.Pxysample(cm, iter=iter)
  } else {
    Pxxy_value <- Pxxy_int(b[[1]]*N, b[[2]]/N, rho)
    Pxyy_value <- Pxyy_int(b[[1]]*N, b[[2]]/N, rho)
    Pxy_value <- Pxy_int(b[[1]]*N, b[[2]]/N, rho)
  }

  var_auc <- (Pxy_value*(1-Pxy_value) +
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
