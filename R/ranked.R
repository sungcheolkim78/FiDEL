# ranked.R - ranked probability based algorithms
#
# Sungcheol Kim @ IBM
# 2020/01/15
#
# version 1.0

#library(pROC)
library(data.table)

# There are 3 methods for AUC calculation
# 1) using class probability at rank
# 2) using area under ROC curve using trapezoid rule
# 3) using combinatorial calculation of Pxy

score.to.classprob <- function(classifier, class1=NULL, N=0, M=100, debug.flag=FALSE) {
  check <- c('score', 'y') %in% names(classifier)
  stopifnot(all(check))

  N.data <- length(classifier$y)
  if (N == 0) { N <- N.data}
  stopifnot(N.data >= N)

  # find class 1 and 2 with alphabetical order
  labels <- levels(classifier$y)
  stopifnot (length(labels) == 2)    # check binary classifier

  res <- matrix(0, nrow=N, ncol=M)

  if (is.null(class1)) {
    class1 <- labels[[1]]
    score.class1 <- mean(classifier$score[classifier$y == class1])
    score.class2 <- mean(classifier$score[classifier$y != class1])
    if (score.class2 < score.class1) {
      class1 <- labels[[2]]
    }
  }

  if (N == N.data) {
    classifier$prob <- as.double(classifier$y == class1)
    classifier$rank <- frankv(classifier$score, order=-1L)        # add rank

    if (debug.flag) {
      plot(classifier$rank, classifier$prob)
    }
    return (classifier)
  } else {
    for (i in 1:M) {
      idx <- sample(1:N.data, N)
      temp <- classifier[idx, ]
      temp <- temp[order(temp$score, decreasing = FALSE), ]
      res[, i] <- as.double(temp$y == class1)
    }
    prob <- rowMeans(res)
    rank <- 1:N

    if (debug.flag) {
      plot(rank, prob)
    }
    return (data.frame(rank=rank, prob=prob))
  }
}

# Using class proability at given rank
auc.rank <- function(score, y=NULL, class1=NULL, full=FALSE) {
  # 1) input data frame : score and label (y)
  # 2) score and label
  # 3) rank and prob

  if (is.null(y)) {
    mat <- score
  } else {
    mat <- data.frame(score=score, y=y)
  }

  check <- c('rank', 'prob') %in% names(mat)
  if (!all(check)) {
    # add rank and probability column
    mat <- score.to.classprob(mat, class1=class1)
  }

  # calculate class 1 and class 2
  N <- length(mat$rank)
  N1 <- sum(mat$prob)
  N2 <- N - N1
  rho <- N1/N

  res <- abs(sum(mat$rank*mat$prob)/N1 - sum(mat$rank*(1-mat$prob))/N2)/N + 0.5
  if (full) {
    return(data.frame(auc=res, N=N, rho=rho))
  } else {
    return(res)
  }
}

confMatrix <- function(score, threshold=0.0, first = TRUE) {
  check <- c('score', 'y') %in% names(score)
  stopifnot(all(check))

  lbs <- levels(score$y)
  if (first) {
    class1 <- lbs[[1]]
    class2 <- lbs[[2]]
  } else {
    class1 <- lbs[[2]]
    class2 <- lbs[[1]]
  }
  score$pred <- score$score < threshold
  A <- sum(score$pred == TRUE & score$y == class1)
  B <- sum(score$pred == TRUE & score$y == class2)
  C <- sum(score$pred == FALSE & score$y == class1)
  D <- sum(score$pred == FALSE & score$y == class2)

  if (FALSE) {
    message(paste0(sum(score$pred == TRUE & score$y == class1), " = A"))
    message(paste0(sum(score$pred == TRUE & score$y == class2), " = B"))
    message(paste0(sum(score$pred == FALSE & score$y == class1), " = C"))
    message(paste0(sum(score$pred == FALSE & score$y == class2), " = D"))
  }

  sens <- A/(A+C)
  spec <- D/(B+D)
  prec <- A/(A+B)
  rec <- A/(A+C)
  auc <- auc.rank(score)
  bacc <- (sens + spec)/2

  return(data.frame(sensitivity=sens, specificity=spec, precision=prec, recall=rec, baccuracy=bacc, auc=auc))
}


# calculate lambda1,2 from auc, rho
lambda.auc <- function(auc, N=100, rho=0.5, full=FALSE) {
  costFunc <- function(l, rho, auroc, N) {
    r <- 1:N
    sum1 <- sum(1/(1+exp(l[2]*r - l[1])))/N
    sum2 <- sum(r/(1+exp(l[2]*r - l[1])))/(N*N*rho)

    (rho - sum1)^2 + (1 + .5/N - rho/2 - auroc*(1-rho) - sum2)^2
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

  l <- lambda.appr(auroc, N=N, rho=rho)
  initial <- c(-l[['l1']], l[['l2']])

  temp <- optim(initial, costFunc, rho=rho, auroc=auroc, N=N)
  if (temp$convergence > 1) message("not converged")
  l1 <- -temp$par[1]
  l2 <- temp$par[2]
  rs <- 1/l2 * log((1 - rho)/rho) - l1/l2

  if (full)
    return(c(l1 = l1, l2 = l2, rs = rs, l10 = initial[[1]], l20 = initial[[2]]))
  else
    return(c(l1 = l1, l2 = l2, rs = rs))
}

# calculate lambda1, lambda2 from auc, rho (version 2)
lambda.appr <- function(auc, N=N, rho=rho) {
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

plot.prediction <- function(pred) {
  check <- c("score", "y", "rank") %in% names(pred)
  stopifnot(all(check))

  ggplot(data=pred, aes(x=score, color=y)) + geom_histogram(alpha=0.5, position="identity", bins=50) +
    theme_classic() +
    theme(legend.position="top")
}

ensemble.gaussian <- function(predictlist, alpha = 1.0, method='+', view = FALSE) {
  M <- length(predictlist)
  N <- length(predictlist[[1]]$score)
  y <- predictlist[[1]]$y

  res <- matrix(0, nrow=N, ncol=M)
  fd <- matrix(0, nrow=M, ncol=4)

  for(m in 1:M) {
    fd[m, 1] <- auc.rank(predictlist[[m]])
    l <- lambda.auc(auc.rank(predictlist[[m]], full=TRUE))
    fd[m, 2] <- l[['l1']]
    fd[m, 3] <- l[['l2']]
    fd[m, 4] <- l[['rs']]

    res[ , m] <- predictlist[[m]][['rank']]
    if (method == '+')
      res[ , m] <- l[['l2']]^alpha *(l[['rs']] - res[, m])
    else
      res[ , m] <- 12*N*(fd[m, 1] - 0.5)/(N*N - 1) *((N+1.)/2.- res[, m])
  }

  if(view) {
    temp <- as.data.frame(res)
    #temp[['summa+']] <- rowMeans(res)
    #plot(temp)
    print(cor(temp, method = "spearman"))
    print(fd)
  }

  res <- data.frame(score=-rowMeans(res), y=y)
  res$rank <- rank(res$score)
  return(res)
}

fermi <- function(r, l1, l2) {
  return (1/(1+exp(l2*r+l1)))
}
