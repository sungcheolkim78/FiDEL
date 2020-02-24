# Simulator.R
#
# Sungcheol Kim @ IBM
# 2020/01/15
#
# version 1.0

library(purrr)

# generate labels - class1 and class2
generate.labels <- function(N=100, rho=0.5) {
  res <- rep('Class2', N)
  idx <- sample(N, N*rho)
  res[idx] <- 'Class1'
  as.factor(res)
}

# binary classifier using Gaussian score distribution
classifier.gaussian <- function(y, auc=0.8, tol=0.0001, max_iter=2000) {
  # check key numbers
  lbs <- levels(y)
  stopifnot(length(lbs) == 2)
  class1 <- lbs[[1]]
  class2 <- lbs[[2]]

  N <- length(y)
  N1 <- sum(y == class1)
  N2 <- N - N1

  count <- 1
  # initial mu2 value
  mu <- 2 * erf.inv(2*auc - 1)

  simulated_auc = 0.5
  while((abs(simulated_auc - auc) > tol) & (count < max_iter)) {
    score1 <- rnorm(N1, mean=0, sd=1)        # class 2 score (lower)
    score2 <- rnorm(N2, mean=mu, sd=1)       # class 1 score (higher)

    score <- rep(0, N)

    score[y == class1] <- score1
    score[y == class2] <- score2

    res <- data.frame(score=score, y=y)

    simulated_auc <- auc.rank(res, class1=class1)
    #simulated_auc <- auc(roc(res$y, res$score))

    count <- count + 1
  }
  msg <- paste0('Final AUC: ', simulated_auc, ' (iter: ', count, ') mu2: ', mu)
  message(msg)

  return(res)
}

generate.auclist <- function(initial, final, N) {
  delta <- (final - initial)/N
  initial + (1:N)*delta - delta
}

generate.ensemble <- function(auclist, N=100, rho=0.5, method='+') {
  y <- generate.class(N=N, rho=rho)
  glist <- purrr::map(auclist, function(x) predict.gaussian(y, x))
  names(glist) <- auclist

  temp <- ensemble.gaussian(glist, method=method)
  glist$summap <- temp
  #print(str(glist))
  glist
}

testsumma <- function(auc0, auc1, method, rho, M) {
  auclist <- generate.auclist(auc0, auc1, M)
  summap <- generate.ensemble(auclist, N=1000, rho=rho, method=method)
  df <- confMatrix(summap$summap)
  df$method <- paste0('summa', method)
  df$aucrange <- paste0(auc0, '-', auc1)
  df$rho <- rho
  df$M <- M

  df
}

erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1
erf.inv <- function(x) qnorm((x + 1)/2)/sqrt(2)
