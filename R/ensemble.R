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
library(ggrepel)
library(tictoc)
library(data.table)

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

setClass("FDensemble",
         representation(predictions = "matrix",
                        logit_matrix = "matrix",
                        rank_matrix = "matrix",
                        beta = "numeric",
                        mu = "numeric",
                        rstar = "numeric",
                        nsamples = "numeric",
                        nmethods = "numeric",
                        method_names = "character",
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

  # set base classifier names
  if (is.null(colnames(predictions))) {
    colnames(predictions) <- 1:.Object@nmethods
  }
  .Object@method_names <- colnames(predictions)
  names(.Object@method_names) <- colnames(predictions)

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
  for (i in seq_along(.Object@actual_performance)) {
    .Object@method_names[i] <- paste0(.Object@method_names[i], '\n', 'A=',
                                      round(.Object@actual_performance[i], digits=3))
  }

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
  .Object@estimated_logit <- -rowSums(.Object@logit_matrix)
  .Object@estimated_label <- as_label(ifelse(.Object@estimated_logit >0, 'class1', 'class2'))
  .Object@estimated_prob <- 1/(1+exp(-.Object@estimated_logit))
  .Object@estimated_rank <- frankv(.Object@estimated_logit)

  if (ncol(.Object@rank_matrix) != ncol(.Object@predictions)) {
    .Object@rank_matrix <- .Object@rank_matrix[, -ncol(.Object@rank_matrix)]
  }
  #colnames(.Object@rank_matrix) <- classifier_names
  .Object@rank_matrix <- cbind(.Object@rank_matrix, A_FD=.Object@estimated_rank)

  # calculate ensemble AUC
  .Object@ensemble_auc <- auc.rank(.Object@estimated_logit, actual_label)
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
  .Object@estimated_performance <- actual_performance

  for (i in seq_along(.Object@actual_performance)) {
    .Object@method_names[i] <- paste0(.Object@method_names[i], '\n', 'A=',
                                      round(.Object@actual_performance[i], digits=3))
  }

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
  .Object@estimated_label <- as_label(ifelse(.Object@estimated_logit >0, 'class1', 'class2'))
  .Object@estimated_prob <- 1/(1+exp(-.Object@estimated_logit))
  .Object@estimated_rank <- frankv(.Object@estimated_logit)
  .Object@ensemble_auc <- auc.rank(.Object@estimated_logit, .Object@estimated_label)

  if (ncol(.Object@rank_matrix) != ncol(.Object@predictions)) {
    .Object@rank_matrix <- .Object@rank_matrix[, -ncol(.Object@rank_matrix)]
  }
  #colnames(.Object@rank_matrix) <- classifier_names
  .Object@rank_matrix <- cbind(.Object@rank_matrix, A_FD=.Object@estimated_rank)

  #cat('... Ensemble AUC:', .Object@ensemble_auc, '\n')
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

setMethod("plot_cor", "FDensemble", function(.Object, filename='cor.pdf', class_flag='all', legend_flag=FALSE) {
  if (length(.Object@actual_label) == 0) {
    colorder <- order(.Object@estimated_performance)
    colorder <- c(colorder, .Object@nmethods+1)
    if (class_flag == 'all') { cor_m <- cor(.Object@rank_matrix) }
    if (class_flag == 'positive') {
      cor_m <- cor(.Object@rank_matrix[.Object@estimated_label == attr(.Object@estimated_label, 'class1'), colorder])
    }
    if (class_flag == 'negative') {
      cor_m <- cor(.Object@rank_matrix[.Object@estimated_label != attr(.Object@estimated_label, 'class1'), colorder])
    }
  } else {
    colorder <- order(.Object@actual_performance)
    colorder <- c(colorder, .Object@nmethods+1)
    if (class_flag == 'all') { cor_m <- cor(.Object@rank_matrix) }
    if (class_flag == 'positive') {
      cor_m <- cor(.Object@rank_matrix[.Object@actual_label == attr(.Object@actual_label, 'class1'), colorder])
    }
    if (class_flag == 'negative') {
      cor_m <- cor(.Object@rank_matrix[.Object@actual_label != attr(.Object@actual_label, 'class1'), colorder])
    }
  }
  cor_m[upper.tri(cor_m)] <- NA
  melted_cor_m <- reshape2::melt(cor_m, na.rm=TRUE)

  g <- ggplot(melted_cor_m, aes(Var1, Var2)) +
    geom_tile(aes(fill = value), colour = "white") +
    scale_fill_gradient2(mid = "white", low = cbPalette[6], high=cbPalette[7], midpoint=0, limit=c(-1,1)) +
    theme_grey(base_size = 10) + labs(x = "", y = "") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(axis.text.x = element_text(angle = 45, vjust = 1,size = 9, hjust = 1)) +
    geom_text(aes(x=Var1, y=Var2, label=round(value, digits = 3)), size=2) +
    ggtitle(class_flag)

  if (!legend_flag) {
    g <- g + theme(legend.position = "none")
  }

  ggsave(filename, g, width=8, height=7)
  print(g)
  #return(cor_m)
})

setGeneric("plot_ensemble", function(.Object, filename='ens.pdf', ...) {standardGeneric("plot_ensemble")})

setMethod("plot_ensemble", "FDensemble", function(.Object, filename='ens.pdf', method='AUC', alpha=0.95, amax=0) {
  # prepare data
  method_list = c('auc', 'asof', 'correlation', 'random', 'invauc')
  if (!(method %in% method_list)) {
    cat('... possible options for method: ', method_list)
    return (FALSE)
  }

  if (method == 'auc') {
    order_index = order(.Object@actual_performance)
  } else if (method == 'invauc') {
    order_index = order(.Object@actual_performance, decreasing = TRUE)
  } else if (method == 'asof') {
    order_index = 1:.Object@nmethods
  } else if (method == 'correlation') {
    order_index = cal_least_cor_list(.Object)
  } else if (method == 'random') {
    order_index = sample(1:.Object@nmethods)
  }

  agg_logit <- t(apply(.Object@logit_matrix[,order_index], 1, cumsum))
  agg_auc <- apply(agg_logit, 2, auc.rank, .Object@actual_label)
  min_auc <- min(.Object@actual_performance)
  max_auc <- max(.Object@actual_performance)
  df <- data.table(x=1:.Object@nmethods, agg_auc=agg_auc, algorithm=.Object@method_names[order_index])

  # check ensemble auc > max base auc
  if (max(agg_auc) > max_auc) {
    label_x = .Object@nmethods
    hjust = 1
  } else {
    label_x = 1
    hjust = 0
  }
  print(df)

  # generate plot
  g <- ggplot(df) + geom_line(aes(x, agg_auc)) +
    geom_point(aes(x,agg_auc)) + theme_classic() +
    geom_label_repel(aes(x=x, y=agg_auc, label=algorithm),
                     size=3.5, fill=cbPalette[3], color='white', segment.color='gray',
                     box.padding = 0.75, alpha=alpha) +
    xlab('Number of methods') + ylab('Aggregated AUC') +
    ggtitle(paste0('Sequentially Aggregated Classifiers ordered by ', method)) +
    geom_hline(yintercept = min_auc, linetype='dashed', color=cbPalette[6], alpha=0.7) +
    geom_hline(yintercept = max_auc, linetype='dashed', color=cbPalette[2], alpha=0.7) +
    annotate(geom='text', x=.Object@nmethods, y=min_auc+0.001, label='min. of base classifiers', hjust=1, vjust=0) +
    annotate(geom='text', x=label_x, y=max_auc-0.001, label='max. of base classifiers', hjust=hjust, vjust=1)

  if (amax > 0) {
    g <- g + ylim(c(min_auc, amax))
  }

  ggsave(filename, width=8, height=6)
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

cal_least_cor_list <- function(.Object) {
  # reorder by AUC
  colorder <- order(.Object@actual_performance, decreasing = TRUE)

  # create correlation matrix
  cor_m <- abs(cor(.Object@rank_matrix[.Object@actual_label == attr(.Object@actual_label, 'class1'),
                                   colorder]))
  #print(cor_m)

  idx_lc <- c(which.max(cor_m[1,]), which.min(cor_m[1,]))
  cor_m <- cor_m[,-idx_lc]
  names_lc <- names(idx_lc)
  for (i in seq(2:(.Object@nmethods-2))) {
    # find least correlated method
    corsum <- colSums(cor_m[idx_lc,])

    j <- which.min(corsum)
    idx_lc <- c(idx_lc, j)

    if (ncol(cor_m) > 2) {
      names_lc <- c(names_lc, names(j))
    } else {
      names_lc <- c(names_lc, names(which.min(corsum)), names(which.max(corsum)))
    }
    cor_m <- cor_m[,-j]
  }

  #print(idx_lc)
  return(names_lc)
}
