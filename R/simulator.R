# simulator.R
#
# Sung-Cheol Kim @ IBM
#
# version 1.0.0 - 2020/01/15
# version 1.0.1 - 2020/04/13 - clean up and check
# version 1.1.0 - 2020/12/28 - add comments

library(data.table)

#' generate labels - class1 and class2
#'
#' @param N size of labels
#' @param rho the prevalence N1/N
#' @return A list of labels
create.labels <- function(N=100, rho=0.5) {
  res <- rep('Class2', N)
  idx <- sample(N, N*rho)
  res[idx] <- 'Class1'
  res <- as.factor(res)

  return (to_label(res))
}

#' add information about labels
#'
#' @param ylist A list of labels
#' @param class1 the label of class 1
#' @param asfactor A boolean value
#' @return A list of labels
to_label <- function(ylist, class1=NULL, asfactor=FALSE) {
  # check y is binary system
  llist <- unique(ylist)
  if (length(llist) > 2) { stop(paste0('... Not binary case: ', llist)) }

  if (is.null(class1)) {
    class1 <- llist[[1]]
    class2 <- llist[[2]]
  } else {
    class2 <- llist[llist != class1]
  }

  attr(ylist, 'class1') <- class1
  attr(ylist, 'class2') <- class2
  attr(ylist, 'N') <- length(ylist)
  attr(ylist, 'N1') <- sum(ylist == class1)
  attr(ylist, 'N2') <- sum(ylist == class2)
  attr(ylist, 'rho') <- sum(ylist == class1)/length(ylist)

  if (asfactor) return (as.factor(ylist))
  else return (ylist)
}

#' binary classifier using Gaussian score distribution
#'
#' @param auc the AUC value
#' @param y A list of labels
#' @param tol the tolerance
#' @param max_iter the maximum iteration to create correct AUC
#' @return A list of scores
create.scores.gaussian <- function(auc0, y, tol=0.0001, max_iter=2000) {
  # check key numbers

  if (is.null(attr(y, 'rho')) || attr(y, 'rho') == 0) { y <- to_label(y) }

  N <- attr(y, 'N')
  N1 <- floor(attr(y, 'N') * attr(y, 'rho'))
  N2 <- N - N1

  count <- 1

  # initial mu2 value
  mu <- 2 * erf.inv(2*auc0 - 1)
  max_iter <- max_iter / ((auc0 - 0.5) * 10)

  # repeat until the measured AUC become the target AUC
  simulated_auc = 0.5
  while((abs(simulated_auc - auc0) > tol) & (count < max_iter)) {
    score1 <- rnorm(N1, mean=0, sd=1)        # class 2 score (lower)
    score2 <- rnorm(N2, mean=mu, sd=1)       # class 1 score (higher)

    score <- rep(0, N)

    score[y == attr(y, 'class1')] <- score1
    score[y == attr(y, 'class2')] <- score2

    simulated_auc <- auc_rank(score, y)

    count <- count + 1
  }

  msg <- paste0('Final AUC: ', round(simulated_auc, digits = 4), ' (iter: ', count, ') mu2: ', round(mu, digits = 4))
  message(msg)

  return(score)
}

#' simple code to generate AUC list between initial and final values
#'
#' @param initial the starting AUC value
#' @param final the ending AUC value
#' @param N the number of AUC list
#' @return A list of AUC values
create.auclist <- function(initial=0.51, final=0.91, N=10) {
  if (initial <= 0.5) { initial = 0.501 }
  if (final >= 1.0) { final = 1.0 }

  delta <- (final - initial)/(N-1)
  return(initial + seq(0, N-1)*delta)
}

#' generate prediction matrix for classifer based on Gaussian score
#'
#' @param n the sample size
#' @param m the number of AUC values
#' @param p the prevalence
#' @param auclist A list of AUC (optional)
#' @param y the list of labels (optional)
#' @return predictions, actual_labels, actual_performance
create_predictions <- function(n=1000, m=20, p=0.6, auclist=NULL, y=NULL) {
  if (is.null(auclist)) {
    auclist <- create.auclist(0.51, 0.99, m)
  } else {
    m <- length(auclist)
  }
  if (is.null(y)) {
    y <- create.labels(N=n, rho=p)
  }

  res <- sapply(auclist, create.scores.gaussian, y, tol=0.0001)

  gen_name <- function(x) { paste0('G', round(x, digits=1)) }
  colnames(res) <- sapply(1:m, gen_name)

  return (list(predictions = res, actual_labels = y, actual_performance = auclist))
}
