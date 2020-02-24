# ci.R - Confidence Interval
#
# Sungcheol Kim @ IBM
# 2020/02/14
#
# version 1.0

# Sampling Methods

countXY <- function(classMatrix, seed=1) {
  set.seed(seed)
  N <- length(classMatrix$prob)
  dices <- runif(N)
  check <- (classMatrix$prob > dices)             # make class 1 or 2 from class probability

  r1 <- sample(classMatrix$rank[check], 1)        # choose a rank from class 1
  r2 <- sample(classMatrix$rank[!check], 1)       # choose a rank from class 2

  # check conditions for P_xy
  if (r1 < r2) {    # rank of class 1 is less than that of class 2
    return (1)
  } else {
    return (0)
  }
}

countXXY <- function(classMatrix, seed=1) {
  set.seed(seed)
  N <- length(classMatrix$prob)
  dices <- runif(N)
  check <- (classMatrix$prob > dices)             # make class 1 or 2 from class probability

  r1 <- sample(classMatrix$rank[check], 2)        # choose a rank from class 1
  r2 <- sample(classMatrix$rank[!check], 1)       # choose a rank from class 2

  # check conditions for P_xxy
  if (max(r1) < r2) {                   # rank of class 2 is the highest rank
    return(1)
  } else {
    return(0)
  }
}

countXYY <- function(classMatrix, seed=1) {
  set.seed(seed)
  N <- length(classMatrix$prob)
  dices <- runif(N)
  check <- (classMatrix$prob > dices)             # make class 1 or 2 from class probability

  r1 <- sample(classMatrix$rank[check], 1)        # choose a rank from class 1
  r2 <- sample(classMatrix$rank[!check], 2)       # choose a rank from class 2

  # check conditions for P_xyy
  if (r1 < min(r2)) {                  # rank of class 1 is the lowest rank
    return (1)
  } else {
    return (0)
  }
}

# Random sampling
auc.Pxysample <- function(classMatrix, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXY(classMatrix, seed=.x))
  res <- cumsum(res)/(1:iter)
  #print(res[length(res)])
  if (debug.flag) plot(res)

  return (res[length(res)])
}

Pxxy.sample <- function(classMatrix, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXXY(classMatrix, seed=.x))
  res <- cumsum(res)/(1:iter)

  if (debug.flag) plot(res)

  return (res[length(res)])
}

Pxyy.sample <- function(classMatrix, iter=5000, debug.flag=FALSE) {
  iters <- sample(1:iter)
  res <- purrr::map_dbl(iters, ~countXYY(classMatrix, seed=.x))
  res <- cumsum(res)/(1:iter)

  if (debug.flag) plot(res)

  return (res[length(res)])
}

# Summation of class probability at rank
auc.Pxysum <- function(classMatrix, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(classMatrix)
  stopifnot(all(check))

  probs <- classMatrix[order(classMatrix$rank, decreasing = FALSE), "prob"]
  res <- outer(probs, 1 - probs, "*")
  idx <- upper.tri(res)

  N <- length(classMatrix$prob)
  N1 <- sum(classMatrix$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(sum(res[idx])/(N1*N2))
}

Pxxy.sum <- function(classMatrix, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(classMatrix)
  stopifnot(all(check))

  res <- 0
  N <- length(classMatrix$prob)
  probs <- classMatrix[order(classMatrix$rank), "prob"]

  for (i in 1:N) {
    for (j in 1:N) {
      for (k in 1:N) {
        if (k > max(c(i, j))) {
          res <- res + probs[i]*probs[j]*(1-probs[k])
        }
      }
    }
  }

  N1 <- sum(classMatrix$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(res/(N1*N1*N2))
}

Pxxy.sum2 <- function(classMatrix, debug.flag=FALSE) {
  check <- c("rank", "prob") %in% names(classMatrix)
  stopifnot(all(check))

  res <- 0
  N <- length(classMatrix$prob)
  probs <- classMatrix[order(classMatrix$rank), "prob"]
  r2r3 <- outer(probs, 1 - probs, "*")

  for (i in 1:N) {
    idx <- upper.tri(r2r3, diag=TRUE)
    idx[, 1:i] <- FALSE

    res <- res + probs[i] * sum(r2r3[idx])
  }

  N1 <- sum(classMatrix$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(res/(N1*N1*N2))
}

Pxyy.sum <- function(classMatrix, debug.flag=FALSE) {
  check <- c("score", "y", "rank", "prob") %in% names(classMatrix)
  stopifnot(all(check))

  N <- length(classMatrix$prob)
  probs <- classMatrix[order(classMatrix$rank), "prob"]

  pxyy <- function(i, j, k) {
    if (i < min(c(j,k))) {
      return (probs[i]*(1-probs[j])*(1-probs[k]))
    }
  }

  res <- purrr::reduce(expand.grid(1:N, 1:N, 1:N), ~pxyy(..1, ..2, ..3))

  N1 <- sum(classMatrix$prob)
  N2 <- N - N1

  return(res/(N1*N2*N2))
}

Pxyy.sum2 <- function(classMatrix, debug.flag=FALSE) {
  check <- c("score", "y", "rank", "prob") %in% names(classMatrix)
  stopifnot(all(check))

  res <- 0
  N <- length(classMatrix$prob)
  probs <- classMatrix[order(classMatrix$rank), "prob"]
  r2r3 <- outer(1 - probs, 1 - probs, "*")

  for (i in 1:N) {
    idx <- (r2r3 < 2)    # make all TRUE
    idx[, 1:i] <- FALSE
    idx[1:i, ] <- FALSE
    diag(idx) <- FALSE

    res <- res + probs[i] * sum(r2r3[idx])
  }

  N1 <- sum(classMatrix$prob)
  N2 <- N - N1

  if (debug.flag) {
    image(res, useRaster=TRUE)
    #image(idx, useRaster=TRUE)
  }

  return(res/(N1*N2*N2))
}

var.auc <- function(classMatrix, method='Sampling', debug.flag=FALSE) {
  N <- length(classMatrix$prob)
  N1 <- sum(classMatrix$prob)
  N2 <- N - N1

  if (method == 'Sampling') {
    Pxxy_value <- Pxxy.sample(classMatrix)
    Pxyy_value <- Pxyy.sample(classMatrix)
    Pxy_value <- auc.Pxysample(classMatrix)
  } else if (method == 'Sum') {
    Pxxy_value <- Pxxy.sum2(classMatrix)
    Pxyy_value <- Pxyy.sum2(classMatrix)
    Pxy_value <- auc.Pxysum(classMatrix)
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

var_auc_fermi <- function(auc, N1, N2, iter=5000, debug.flag=FALSE) {
  N <- N1 + N2
  l <- lambda.auc(auc, N=N, rho=N1/N)
  if (debug.flag) print(l)

  ranks <- 1:N
  probs <- fermi(ranks, l['l1'], l['l2'])
  if (debug.flag) plot(ranks, probs)
  cm <- data.frame(rank=ranks, prob=probs)

  Pxxy_value <- Pxxy.sample(cm, iter=iter)
  Pxyy_value <- Pxyy.sample(cm, iter=iter)
  Pxy_value <- auc.Pxysample(cm, iter=iter)
  var_auc <- (Pxy_value*(1-Pxy_value) +
                (N1 - 1)*(Pxxy_value - Pxy_value*Pxy_value) +
                (N2 - 1)*(Pxyy_value - Pxy_value*Pxy_value))/(N1*N2)

  if (debug.flag) {
    print(paste0('Rho: ', N1/N))
    print(paste0('Pxxy: ', Pxxy_value))
    print(paste0('Pxyy: ', Pxyy_value))
    print(paste0('Pxy: ', Pxy_value))
    print(paste0('AUC: ', auc))
    print(paste0('AUC Sigma: ', sqrt(var_auc)))
    print(paste0('95% CI: ', auc - 1.96*sqrt(var_auc), ' - ', auc + 1.96*sqrt(var_auc)))
  }
  return(var_auc)
}
