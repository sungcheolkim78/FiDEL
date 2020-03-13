# ensemble.R - ensemble object using FD distribution
#
# Sungcheol Kim @ IBM
# 2020/03/10
#
# version 1.0

library(ggplot2)
library(ggpubr)

setClass("FDensemble",
         representation(predictions = "matrix",
                        rank_matrix = "matrix",
                        lambda1s = "numeric",
                        lambda2s = "numeric",
                        rstars = "numeric",
                        nsamples = "numeric",
                        nmethods = "numeric",
                        estimated_label = "numeric",
                        actual_label = "numeric",
                        ensemble_auc = "numeric",
                        actual_performance = "numeric",
                        estimated_performance = "numeric",
                        estimated_prevalence = "numeric",
                        actual_prevalence = "numeric",
                        estimated_rank = "numeric"))

setMethod("initialize", "FDensemble", function(.Object, predictions) {
  cat("... FDensemble initializer \n")
  .Object@predictions <- predictions
  .Object@nsamples <- nrow(predictions)
  .Object@nmethods <- ncol(predictions)



  return(.Object)
})

fde <- fdensemble <- function(predictions, actual_performance) {
  cat("... FDensemble constructor \n")
  if (missing(actual_performance)) { new (Class="FDensemble", predictions = predictions) }
  else {
    obj <- new (Class="FDensemble", predictions = predictions, actual_performance = actual_performance)
    calculate_performance(obj, actual_performance)
  }
}

setMethod("show", "FDensemble", function(object) {
  cat("# of methods: ", object@nmethods, "\n",
      "# of samples: ", object@nsamples)
})

setValidity("FDensemble", function(object) {
  if (object@nsamples == dim(object@predictions)[1]) {
    return (TRUE)
  } else {
    return ("predictions should have dimensions as nsamples x nmethods")
  }
})

setGeneric("calculate_performance", function(.Object, actual_labels) {standardGeneric("calculate_performance")})

setMethod("calculate_performance", "FDensemble", function(.Object, actual_labels) {
  # save label data
  .Object@actual_label <- actual_labels

  # calculate auc using labels
  .Object@actual_performance <- apply(.Object@predictions, 2, auc.rank, actual_labels)

  # calculate lambda1, 2, and rstar
  .Object@actual_prevalence <- attr(actual_labels, 'rho')
  l_list <- sapply(.Object@actual_performance, lambda.auc, N=.Object@nsamples, rho=.Object@actual_prevalence)
  .Object@lambda1s <- as.vector(l_list[1, ])
  .Object@lambda2s <- as.vector(l_list[2, ])
  .Object@rstars <- as.vector(l_list[3, ])

  # calculate new rank scores
  res <- matrix(0, nrow=.Object@nsamples, ncol=.Object@nmethods)
  for (m in seq(1, .Object@nmethods)) {
    res[ , m] <- rank(.Object@predictions[, m])
    res[ , m] <- .Object@lambda2s[m] *(.Object@rstars[m] - res[, m])
  }
  .Object@rank_matrix <- res
  .Object@estimated_rank <- rowSums(res)

  # calculate ensemble AUC
  .Object@ensemble_auc <- auc.rank(.Object@estimated_rank, actual_labels)

  return(.Object)
})

setGeneric("predict_performance", function(.Object, actual_performance, p, alpha=1) {standardGeneric("predict_performance")})

setMethod("predict_performance", "FDensemble", function(.Object, actual_performance, p, alpha=1) {
  # calculate auc using labels
  .Object@actual_performance <- actual_performance

  # calculate lambda1, 2, and rstar
  .Object@actual_prevalence <- p
  l_list <- sapply(.Object@actual_performance, get_fermi, .Object@actual_prevalence)
  .Object@lambda1s <- as.vector(-l_list[1, ]*l_list[2, ])
  .Object@lambda2s <- as.vector(l_list[1, ]/.Object@nsamples)
  .Object@rstars <- as.vector(l_list[3, ]*.Object@nsamples)

  # calculate new rank scores
  res <- matrix(0, nrow=.Object@nsamples, ncol=.Object@nmethods)
  for (m in seq(1, .Object@nmethods)) {
    res[ , m] <- rank(.Object@predictions[, m])
    res[ , m] <- .Object@lambda2s[m]^alpha *(.Object@rstars[m] - res[, m])
  }
  .Object@rank_matrix <- res
  .Object@estimated_rank <- rowSums(res)

  return(.Object)
})

setGeneric("plot_FDstatistics", function(.Object) {standardGeneric("plot_FDstatistics")})

setMethod("plot_FDstatistics", "FDensemble", function(.Object) {
    df <- data.frame(x=.Object@actual_performance, l1=.Object@lambda1s, l2=.Object@lambda2s, rs=.Object@rstars)
    g1 <- ggplot(data = df) + geom_point(aes(x=x, y=l1)) + xlab('AUC') + ylab('Lambda_1') + theme_classic()
    g2 <- ggplot(data = df) + geom_point(aes(x=x, y=l2)) + xlab('AUC') + ylab('Lambda_2') + theme_classic()
    g3 <- ggplot(data = df) + geom_point(aes(x=x, y=rs)) + xlab('AUC') + ylab('R_star') + theme_classic()

    g <- ggarrange(g1, g2, g3, labels=c("A", "B", "C"), ncol=3  , nrow=1)
    print(g)
})

setGeneric("plot_single", function(.Object, target, ...) {standardGeneric("plot_single")})

setMethod("plot_single", "FDensemble", function(.Object, target, c, n=100, m=100) {
  if (missing(c)) { c <- 1 }
  if (c > .Object@nmethods) { c <- .Object@nmethods }

  if (target == 'score') {
    g <- plot.scores(.Object@predictions[ , c], .Object@actual_label)
    print(g)
  }

  if (target == 'roc') {
    pcr1 <- pcr(.Object@predictions[ , c], .Object@actual_label, N=n, M=m)
    g <- plot.curves(pcr1)
    print(g)
  }

  if (target == 'pcr') {
    pcr1 <- pcr(.Object@predictions[ , c], .Object@actual_label, N=n, M=m)
    g <- plot.pcr(pcr1)
    print(g)
  }
})
