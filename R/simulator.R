# Simulator.R
#
# Sungcheol Kim @ IBM
# 2020/01/15
#
# version 1.0

library(purrr)
library(data.table)

# generate labels - class1 and class2
create.labels <- function(N=100, rho=0.5) {
  res <- rep('Class2', N)
  idx <- sample(N, N*rho)
  res[idx] <- 'Class1'
  res <- as.factor(res)
  attr(res, 'class1') <- 'Class1'
  attr(res, 'class2') <- 'Class2'
  attr(res, 'N') <- N
  attr(res, 'rho') <- rho
  attr(res, 'class') <- c('factor', 'label')

  return(res)
}

as_label <- function(ylist, class1=NULL) {
  if (any('label' %in% class(ylist))) return(ylist)
  ylist <- as.factor(ylist)

  llist <- labels(ylist)
  print(llist)
  if (is.null(class1)) {
    class1 <- llist[[1]]
    class2 <- llist[[2]]
  } else {
    class2 <- llist[llist != class1]
  }
  print(class1)
  print(class2)

  attr(ylist, 'class1') <- class1
  attr(ylist, 'class2') <- class2
  attr(ylist, 'N') <- length(ylist)
  attr(ylist, 'rho') <- sum(ylist == class1)/length(ylist)
  attr(ylist, 'class') <- c('factor', 'label')

  return(ylist)
}

# binary classifier using Gaussian score distribution
create.scores.gaussian <- function(y, auc=0.8, tol=0.0001, max_iter=2000) {
  # check key numbers

  if (any('label' == class(y))) {
    N1 <- floor(attr(y, 'N') * attr(y, 'rho'))
    N2 <- attr(y, 'N') - N1
    N <- N1 + N2
    class1 <- attr(y, 'class1')
    class2 <- attr(y, 'class2')
  } else {
    lbs <- levels(y)
    stopifnot(length(lbs) == 2)
    class1 <- lbs[[1]]
    class2 <- lbs[[2]]

    N <- length(y)
    N1 <- sum(y == class1)
    N2 <- N - N1
  }

  count <- 1
  # initial mu2 value
  mu <- 2 * erf.inv(2*auc - 1)

  # repeat until the measured AUC become the target AUC
  simulated_auc = 0.5
  while((abs(simulated_auc - auc) > tol) & (count < max_iter)) {
    score1 <- rnorm(N1, mean=0, sd=1)        # class 2 score (lower)
    score2 <- rnorm(N2, mean=mu, sd=1)       # class 1 score (higher)

    score <- rep(0, N)

    score[y == class1] <- score1
    score[y == class2] <- score2

    simulated_auc <- auc.rank(score, y)

    count <- count + 1
  }
  msg <- paste0('Final AUC: ', simulated_auc, ' (iter: ', count, ') mu2: ', mu)
  message(msg)

  return(score)
}

# simple code to generate AUC list between initial and final values
create.auclist <- function(initial, final, N) {
  if (initial <= 0.5) { initial = 0.501 }
  if (final >= 1.0) { final = 1.0 }

  delta <- (final - initial)/N
  initial + (0:N)*delta
}

# generate classifer based on SUMMA and SUMMA+ method
create_predictions <- function(n=1000, m=30, p=0.6, auclist=NULL, y=NULL, method='rank') {
  if (is.null(auclist)) {
    auclist <- create.auclist(0.51, 0.99, m)
  } else {
    m <- length(auclist)
  }
  if (is.null(y)) {
    y <- create.labels(N=n, rho=p)
  }

  res <- matrix(nrow=n, ncol=m)
  i <- 1

  for (a in auclist) {
    gs <- create.scores.gaussian(y, auc=a, tol=0.0001, max_iter=1000)
    res[ , i] <- gs
    i <- i + 1
  }

  gen_name <- function(x) { paste0('A_', round(x, digits=2)) }
  colnames(res) <- sapply(auclist, gen_name)

  return (list(predictions = res, actual_labels = y, actual_performance = auclist))
}
