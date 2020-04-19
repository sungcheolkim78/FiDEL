# utility.R - utility functions
#
# Sungcheol Kim @ IBM
# 2020/03/12
#
# version 1.0

erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1

erf.inv <- function(x) qnorm((x + 1)/2)/sqrt(2)

fermi.l <- function(x, l1, l2) 1/(1+exp(l2 * x + l1))

fermi.b <- function(x, b, m, normalized=FALSE) {
  if (normalized) {
    return (1/(1 + exp(b/length(x)*(x - m*length(x)))))
  } else {
    return (1/(1 + exp(b*(x - m))))
  }
}

# not used
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

  return(data.table(sensitivity=sens, specificity=spec, precision=prec, recall=rec, baccuracy=bacc, auc=auc))
}
