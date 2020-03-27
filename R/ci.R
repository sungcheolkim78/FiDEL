# ci.R - Confidence Interval
#
# Sungcheol Kim @ IBM
# 2020/02/14
#
# version 1.0
# version 1.1 - using normalized beta, mu

# Sampling Methods
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

# Summation of class probability at rank
auc.Pxysum <- function(pcr_m, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(pcr_m)
  stopifnot(all(check))

  probs <- pcr_m[order(pcr_m$rank, decreasing = FALSE), "prob"]
  res <- outer(probs, 1 - probs, "*")
  idx <- upper.tri(res)

  N <- length(pcr_m$prob)
  N1 <- sum(pcr_m$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(sum(res[idx])/(N1*N2))
}

Pxxy.sum <- function(pcr_m, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(pcr_m)
  stopifnot(all(check))

  res <- 0
  N <- length(pcr_m$prob)
  probs <- pcr_m[order(pcr_m$rank), "prob"]

  for (i in 1:N) {
    for (j in 1:N) {
      for (k in 1:N) {
        if (k > max(c(i, j))) {
          res <- res + probs[i]*probs[j]*(1-probs[k])
        }
      }
    }
  }

  N1 <- sum(pcr_m$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(res/(N1*N1*N2))
}

Pxxy.sum2 <- function(pcr_m, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(pcr_m)
  stopifnot(all(check))

  res <- 0
  N <- length(pcr_m$prob)
  probs <- pcr_m[order(pcr_m$rank), "prob"]
  r2r3 <- outer(probs, 1 - probs, "*")

  for (i in 1:N) {
    idx <- upper.tri(r2r3, diag=TRUE)
    idx[, 1:i] <- FALSE

    res <- res + probs[i] * sum(r2r3[idx])
  }

  N1 <- sum(pcr_m$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(res/(N1*N1*N2))
}

Pxyy.sum <- function(pcr_m, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(pcr_m)
  stopifnot(all(check))

  N <- length(pcr_m$prob)
  probs <- pcr_m[order(pcr_m$rank), "prob"]

  pxyy <- function(i, j, k) {
    if (i < min(c(j,k))) {
      return (probs[i]*(1-probs[j])*(1-probs[k]))
    }
  }

  res <- purrr::reduce(expand.grid(1:N, 1:N, 1:N), ~pxyy(..1, ..2, ..3))

  N1 <- sum(pcr_m$prob)
  N2 <- N - N1

  return(res/(N1*N2*N2))
}

Pxyy.sum2 <- function(pcr_m, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(pcr_m)
  stopifnot(all(check))

  res <- 0
  N <- length(pcr_m$prob)
  probs <- pcr_m[order(pcr_m$rank), "prob"]
  r2r3 <- outer(1 - probs, 1 - probs, "*")

  for (i in 1:N) {
    idx <- (r2r3 < 2)    # make all TRUE
    idx[, 1:i] <- FALSE
    idx[1:i, ] <- FALSE
    diag(idx) <- FALSE

    res <- res + probs[i] * sum(r2r3[idx])
  }

  N1 <- sum(pcr_m$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(res/(N1*N2*N2))
}

# Confidence interval calculation
var.auc <- function(pcr_m, method='Sampling', debug.flag=FALSE) {
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
  res <- sum(A * B) + 2*sum(ptable^2 * (1-ptable)) - sum(ptable^2) - 2*sum(ptable*(1-ptable))
  res <- res * resol^3 / (rho^2 * (1-rho))
  toc()

  return(res)
}

Pxyy_int <- function(beta, mu, rho, resol=1e-6) {
  tic(sprintf('... Pxyy integral calculation with res = %.3g', resol))
  ptable <- fermi.b(seq(0,1,by=resol), beta, mu)

  A <- ptable
  B <- rev(cumsum(rev(1-ptable)))^2
  res <- sum(A * B) + 2*sum(ptable * (1-ptable)^2) - sum((1-ptable)^2) - 2*sum(ptable*(1-ptable))
  res <- res * resol^3 / (rho * (1-rho)^2)
  toc()

  return(res)
}
