# utility.R - utility functions
#
# Sung-Cheol Kim @ IBM
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
