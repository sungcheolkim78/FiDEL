# ensemble.R - ensemble object using FD distribution
#
# Sungcheol Kim @ IBM
# 2020/03/10
#
# version 1.0
# version 1.1 - revision with beta and mu
# version 1.2 - 2020/3/26 - correlation plot

library(ggplot2)
library(ggpubr)
library(tictoc)

setClass("FDensemble",
         representation(predictions = "matrix",
                        logit_matrix = "matrix",
                        rank_matrix = "matrix",
                        beta = "numeric",
                        mu = "numeric",
                        rstar = "numeric",
                        nsamples = "numeric",
                        nmethods = "numeric",
                        ensemble_auc = "numeric",
                        actual_performance = "numeric",
                        actual_prevalence = "numeric",
                        actual_label = "numeric",
                        estimated_performance = "numeric",
                        estimated_prevalence = "numeric",
                        estimated_logit = "numeric",
                        estimated_rank = "numeric",
                        estimated_label = "factor",
                        estimated_prob = "numeric"
                        ))

setMethod("initialize", "FDensemble", function(.Object, predictions, ...) {
  #cat("... FDensemble initializer \n")
  .Object@predictions <- predictions
  .Object@nsamples <- nrow(predictions)
  .Object@nmethods <- ncol(predictions)
  .Object@logit_matrix <- apply(predictions, 2, frankv)
  .Object@rank_matrix <- apply(predictions, 2, frankv)
  .Object@estimated_label <- as.factor(c('class1', rep('class2', .Object@nsamples-1)))

  return(.Object)
})

fde <- fdensemble <- function(predictions, actual_label) {
  obj <- new (Class="FDensemble", predictions = predictions)
  if (!missing(actual_label)) {
    message('... FDensemble with actual_label')
    calculate_performance(obj, actual_label)
  } else {
    message('... FDensemble with only predictions')
    obj
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

setGeneric("calculate_performance", function(.Object, actual_label, alpha=1) {standardGeneric("calculate_performance")})

setMethod("calculate_performance", "FDensemble", function(.Object, actual_label, alpha=1) {
  tic(sprintf('... N:%d, M:%d', .Object@nsamples, .Object@nmethods))
  # save label data
  .Object@actual_label <- actual_label

  # calculate auc using labels
  .Object@actual_performance <- apply(.Object@predictions, 2, auc.rank, actual_label)

  # calculate beta, mu and rstar
  .Object@actual_prevalence <- attr(actual_label, 'rho')
  b_list <- sapply(.Object@actual_performance, get_fermi, rho=.Object@actual_prevalence)
  .Object@beta <- as.vector(b_list[1, ])
  .Object@mu <- as.vector(b_list[2, ])
  .Object@rstar <- as.vector(b_list[3, ])

  # calculate new rank scores
  colnames(.Object@logit_matrix) <- colnames(.Object@predictions)
  .Object@logit_matrix <- apply(.Object@predictions, 2, frankv)/.Object@nsamples
  for (m in seq(1, .Object@nmethods)) {
    .Object@logit_matrix[, m] <- .Object@beta[m]^alpha *(.Object@rstar[m] - .Object@logit_matrix[, m])
  }
  .Object@estimated_rank <- rowSums(.Object@logit_matrix)
  .Object@estimated_label[.Object@estimated_rank > 0] <- 'class1'
  .Object@estimated_prob <- 1/(1+exp(-.Object@estimated_rank))

  # calculate ensemble AUC
  .Object@ensemble_auc <- auc.rank(.Object@estimated_rank, actual_label)
  cat('... Ensemble AUC:', .Object@ensemble_auc, '\n')
  cat('... AUC list:', .Object@actual_performance, '\n')
  cat('... beta list:', .Object@beta, '\n')
  cat('... mu list:', .Object@mu, '\n')
  toc()

  return(.Object)
})

setGeneric("predict_performance", function(.Object, actual_performance, p, alpha=1) {standardGeneric("predict_performance")})

setMethod("predict_performance", "FDensemble", function(.Object, actual_performance, p, alpha=1) {
  tic(sprintf('... N:%d, M:%d', .Object@nsamples, .Object@nmethods))
  # calculate auc using labels
  .Object@actual_performance <- actual_performance
  gen_name <- function(x) { paste0('A_', round(x, digits=3)) }
  classifier_names <- sapply(actual_performance, gen_name)
  colnames(.Object@predictions) <- classifier_names

  # calculate beta, mu and rstar
  .Object@actual_prevalence <- p
  b_list <- sapply(actual_performance, get_fermi, rho=p)
  .Object@beta <- as.vector(b_list[1, ])
  .Object@mu <- as.vector(b_list[2, ])
  .Object@rstar <- as.vector(b_list[3, ])

  # calculate new rank scores
  colnames(.Object@logit_matrix) <- colnames(.Object@predictions)
  .Object@logit_matrix <- apply(.Object@predictions, 2, frankv)/.Object@nsamples
  for (m in seq(1, .Object@nmethods)) {
    .Object@logit_matrix[, m] <- .Object@beta[m]^alpha * (.Object@rstar[m] - .Object@logit_matrix[, m] )
  }
  .Object@estimated_logit <- -rowMeans(.Object@logit_matrix)
  .Object@estimated_label <- as.factor(ifelse(.Object@estimated_logit >0, 'class1', 'class2'))
  .Object@estimated_prob <- 1/(1+exp(-.Object@estimated_logit))
  .Object@estimated_rank <- frankv(.Object@estimated_logit)

  if (ncol(.Object@rank_matrix) != ncol(.Object@predictions)) {
    .Object@rank_matrix <- .Object@rank_matrix[, -ncol(.Object@rank_matrix)]
  }
  colnames(.Object@rank_matrix) <- classifier_names
  .Object@rank_matrix <- cbind(.Object@rank_matrix, A_FD=.Object@estimated_rank)

  cat('... AUC list:', .Object@actual_performance, '\n')
  cat('... beta list:', .Object@beta, '\n')
  cat('... mu list:', .Object@mu, '\n')
  toc()

  return(.Object)
})

setGeneric("plot_FDstatistics", function(.Object) {standardGeneric("plot_FDstatistics")})

setMethod("plot_FDstatistics", "FDensemble", function(.Object) {
    df <- data.frame(x=.Object@actual_performance, b=.Object@beta, m=.Object@mu, rs=.Object@rstar)
    g1 <- ggplot(data = df) + geom_point(aes(x=x, y=b)) + xlab('AUC') + ylab('beta') + theme_classic()
    g2 <- ggplot(data = df) + geom_point(aes(x=x, y=m)) + xlab('AUC') + ylab('mu') + theme_classic()
    g3 <- ggplot(data = df) + geom_point(aes(x=x, y=rs)) + xlab('AUC') + ylab('R_star') + theme_classic()

    g <- ggarrange(g1, g2, g3, labels=c("A", "B", "C"), ncol=3  , nrow=1)
    print(g)
})

setGeneric("plot_cor", function(.Object, filename='cor.pdf', ...) {standardGeneric("plot_cor")})

setMethod("plot_cor", "FDensemble", function(.Object, filename='cor.pdf', legend_flag=FALSE) {
  cor_m <- as.data.table(cor(.Object@rank_matrix))
  cor_m$Var1 <- colnames(.Object@rank_matrix)
  melted_cor_m <- melt(cor_m, id='Var1', variable.name = 'Var2')

  g <- ggplot(melted_cor_m, aes(Var1, Var2)) +
    geom_tile(aes(fill = value), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    theme_grey(base_size = 9) + labs(x = "", y = "") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0))
  if (!legend_flag) {
    g <- g + theme(legend.position = "none")
  }

  ggsave(filename, g, width=8, height=7)
  print(g)
})

setGeneric("plot_single", function(.Object, target, ...) {standardGeneric("plot_single")})

setMethod("plot_single", "FDensemble", function(.Object, target, c, n=100, m=100) {
  if (missing(c)) { c <- 0 }
  if (c > .Object@nmethods) { c <- .Object@nmethods }
  if (c < 0) { c <- 0 }

  if (target == 'score') {
    if (c == 0) { scores <- .Object@estimated_logit }
    else { scores <- .Object@predictions[ , c] }
    if (length(.Object@actual_label) > 0) {
      g <- plot.scores(scores, .Object@actual_label)
    } else {
      g <- plot.scores(scores)
    }
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
