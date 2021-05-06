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

setClass("FiDEL",
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

setMethod("initialize", "FiDEL", function(.Object, predictions, ...) {
  #cat("... FiDEL initializer \n")
  .Object@predictions <- predictions
  .Object@nsamples <- nrow(predictions)
  .Object@nmethods <- ncol(predictions)

  # set base classifier names
  if (is.null(colnames(predictions))) {
    colnames(predictions) <- 1:.Object@nmethods
  }
  .Object@method_names <- colnames(predictions)
  names(.Object@method_names) <- colnames(predictions)

  #if ties allowed
  #.Object@logit_matrix <- apply(predictions, 2, frankv, ties.method="random")
  #.Object@rank_matrix <- apply(predictions, 2, frankv, ties.method="random")

  .Object@logit_matrix <- apply(predictions, 2, frankv, ties.method="random")
  .Object@rank_matrix <- apply(predictions, 2, frankv, ties.method="random")
  .Object@estimated_label <- as.factor(c('class1', rep('class2', .Object@nsamples-1)))

  return(.Object)
})

fde <- FiDEL <- function(predictions, actual_label) {
  obj <- new (Class="FiDEL", predictions = predictions)
  if (!missing(actual_label)) {
    message('... FiDEL with actual_label')
    calculate_performance(obj, actual_label)
  } else {
    message('... FiDEL with only predictions')
    obj
  }
}

setMethod("show", "FiDEL", function(object) {
  cat("# of methods: ", object@nmethods, "\n",
      "# of samples: ", object@nsamples)
})

setValidity("FiDEL", function(object) {
  if (object@nsamples == dim(object@predictions)[1]) {
    return (TRUE)
  } else {
    return ("predictions should have dimensions as nsamples x nmethods")
  }
})

setGeneric("calculate_performance", function(.Object, actual_label, ensemble_method="FiDEL", alpha=1) {standardGeneric("calculate_performance")})

setMethod("calculate_performance", "FiDEL", function(.Object, actual_label, ensemble_method="FiDEL", alpha=1) {
  tic(sprintf('... N:%d, M:%d', .Object@nsamples, .Object@nmethods))
  # save label data

  .Object@actual_label <- actual_label

  # calculate auc using labels
  .Object@actual_performance <- apply(.Object@predictions, 2, auc_rank, actual_label)

  for (i in seq_along(.Object@actual_performance)) {
    .Object@method_names[i] <- paste0(.Object@method_names[i], '\n', 'A=',
                                      round(.Object@actual_performance[i], digits=3))
  }

  .Object@actual_performance[round(.Object@actual_performance, 3)==.500] <- .501

  #rescaling of fidel parameters 
  if (ensemble_method=="FiDEL"){
    .Object@beta <- .Object@beta / dim(.Object@predictions)[1]
  } 

  if (ensemble_method=="WoC"){
    .Object@beta <- rep(1, length(.Object@beta))
  }

  .Object@mu <- .Object@mu * dim(.Object@predictions)[1]
  .Object@rstar <- .Object@rstar * dim(.Object@predictions)[1]

  # calculate new rank scores
  colnames(.Object@logit_matrix) <- colnames(.Object@predictions)
  .Object@logit_matrix <- apply(.Object@predictions, 2, frankv)
  
  for (m in seq(1, .Object@nmethods)) {
    .Object@logit_matrix[, m] <- .Object@beta[m]^alpha *(.Object@rstar[m] - .Object@logit_matrix[, m])
  }

  .Object@estimated_logit <- -rowSums(.Object@logit_matrix)

  .Object@estimated_prob <- 1/(1+exp(-.Object@estimated_logit))

  #if ties allowed
  #.Object@estimated_rank <- frankv(.Object@estimated_logit, ties.method='random')
  .Object@estimated_rank <- frankv(.Object@estimated_logit)


  if (ncol(.Object@rank_matrix) != ncol(.Object@predictions)) {
    .Object@rank_matrix <- .Object@rank_matrix[, -ncol(.Object@rank_matrix)]
  }
  #colnames(.Object@rank_matrix) <- classifier_names
  .Object@rank_matrix <- cbind(.Object@rank_matrix, A_FD=.Object@estimated_rank)

  # calculate ensemble AUC
  .Object@ensemble_auc <- auc_rank(.Object@estimated_logit, actual_label)
  cat('... Ensemble AUC:', .Object@ensemble_auc, '\n')
  cat('... AUC list:', .Object@actual_performance, '\n')
  cat('... beta list:', .Object@beta, '\n')
  cat('... mu list:', .Object@mu, '\n')
  toc()

  return(.Object)
})


#not loading when sourcing R for some reason!!
setGeneric("predict_performance", function(.Object, actual_performance, p, nrow.train, alpha=1) {standardGeneric("predict_performance")})

setMethod("predict_performance", "FiDEL", function(.Object, actual_performance, p, nrow.train, alpha=1) {
  tic(sprintf('... N:%d, M:%d', .Object@nsamples, .Object@nmethods))

  # calculate auc using labels
  .Object@actual_performance <- actual_performance
  .Object@estimated_performance <- actual_performance

  for (i in seq_along(.Object@actual_performance)) {
    .Object@method_names[i] <- paste0(.Object@method_names[i], '\n', 'A=',
                                      round(.Object@actual_performance[i], digits=3))
  }

  #any value exactly .5 is set to .501 for numerical stability issues
  .Object@actual_performance[round(.Object@actual_performance, 3)==.500] <- .501

  .Object@actual_prevalence <- mean(p)

  b_list <- unlist(lapply(1:.Object@nmethods, function(x) get_fermi(.Object@actual_performance[[x]], rho=p[x], N=1)))

  b_listdataframe <- do.call(rbind, b_list %>% as.list()) %>% as.data.frame()
  b_listdataframe$name <- rownames(b_listdataframe)
  b_listdataframe <- b_listdataframe %>% as_tibble() 

  .Object@beta <- b_listdataframe %>% as_tibble() %>% filter(grepl("beta", name)) %>% pull(V1) 
  .Object@mu <- b_listdataframe %>% as_tibble() %>% filter(grepl("mu", name)) %>% pull(V1) 
  .Object@rstar <- b_listdataframe %>% as_tibble() %>% filter(grepl("rs", name)) %>% pull(V1) 

  #cat('... Ensemble AUC:', .Object@ensemble_auc, '\n')
  cat('... AUC list:', .Object@actual_performance, '\n')
  cat('... beta list:', .Object@beta, '\n')
  cat('... mu list:', .Object@mu, '\n')
  toc()

  return(.Object)
})

setGeneric("plot_FDstatistics", function(.Object) {standardGeneric("plot_FDstatistics")})

setMethod("plot_FDstatistics", "FiDEL", function(.Object) {
    df <- data.frame(x=.Object@actual_performance, b=.Object@beta, m=.Object@mu, rs=.Object@rstar)
    g1 <- ggplot(data = df) + geom_point(aes(x=x, y=b)) + xlab('AUC') + ylab('beta') + theme_classic()
    g2 <- ggplot(data = df) + geom_point(aes(x=x, y=m)) + xlab('AUC') + ylab('mu') + theme_classic()
    g3 <- ggplot(data = df) + geom_point(aes(x=x, y=rs)) + xlab('AUC') + ylab('R_star') + theme_classic()

    g <- ggarrange(g1, g2, g3, labels=c("A", "B", "C"), ncol=3  , nrow=1)
    print(g)
})

setGeneric("plot_cor", function(.Object, filename='cor.pdf', ...) {standardGeneric("plot_cor")})

setMethod("plot_cor", "FiDEL", function(.Object, filename='cor.pdf', class_flag='all', legend_flag=FALSE) {
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

setMethod("plot_ensemble", "FiDEL", function(.Object, filename='ens.pdf', method='auc', alpha=0.95, amax=0) {
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
  agg_auc <- apply(agg_logit, 2, auc_rank, .Object@actual_label)
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

setMethod("plot_single", "FiDEL", function(.Object, target, c, n=100, m=100) {
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
    pcr1 <- pcr(.Object@predictions[ , c], .Object@actual_label, sample_size=n, sample_n=m)
    g <- plot.curves(pcr1)
    print(g)
  }

  if (target == 'pcr') {
    pcr1 <- pcr(.Object@predictions[ , c], .Object@actual_label, sample_size=n, sample_n=m)
    g <- plot.pcr(pcr1)
    print(g)
  }
})

setGeneric("plot_performance", function(.Object, ...) {standardGeneric("plot_performance")})

setMethod("plot_performance", "FiDEL", function(.Object, nmethod_list=5:7, nsample=20, trendline=FALSE, filename='FiDEL_perf.pdf') {
  df <- cal_partial_performance(.Object, nmethod_list=nmethod_list, nsample=nsample)
  df$nmethod <- as.factor(df$nmethod)

  x_all <- max(.Object@actual_performance)
  y_all <- .Object@ensemble_auc
  y_final <- mean(y_all)
  min_y <- min(c(df$Best_Indv, df$FiDEL, x_all, y_all))
  max_y <- max(c(df$Best_Indv, df$FiDEL, x_all, y_all))

  g <- ggplot(df, aes(x=Best_Indv, y=FiDEL)) + theme_classic() +
    geom_point() + facet_wrap(~nmethod) +
    xlab('Best AUC from random sampled methods') + xlim(c(min_y, max_y)) +
    ylab('FiDEL AUC') + ylim(c(min_y, max_y)) +
    geom_abline(slope=1, linetype='dashed', alpha=0.7)
    #annotate(geom="curve", x=x_all, y=y_all, xend=x_all, yend=x_all, curvature=-.3, arrow = arrow(length = unit(2, "mm"))) +
    #annotate(geom="text", x=max_y, y=x_all, label='All', hjust=0)
  if (trendline)
    g <- g + geom_smooth(method=loess)

  ggsave(filename, width=11, height=4)
  return (g)
})

plot_performance_nmethods <- function(.Object, nmethod_list=5:7, nsample=20, seed=100, method='SE', filename='FiDEL_perf_nmethod.pdf') {
  df <- cal_partial_performance(.Object, nmethod_list=nmethod_list, nsample=nsample, seed=seed)
  df <- melt(df, id.vars = 'nmethod', variable.name='method', value.name='AUC')

  tmp <- df %>% group_by(nmethod, method) %>%
    mutate(Performance=mean(AUC)) %>%
    mutate(sd=sd(AUC)) %>%
    mutate(N=length(AUC))

  if (method == 'SE') {
    tmp$sd <- tmp$sd/sqrt(tmp$N)
  }

  #tmp$ci <- tmp$se * qt(conf.interval/2 + .5, tmp$N-1)
  tmp$shape <- ifelse(tmp$method == 'FiDEL', "21", "23")

  g <- ggplot(tmp, aes(x=nmethod, y=Performance)) + theme_classic() +
    geom_line(aes(linetype=method, color=method), size=2) +
    geom_errorbar(width=.1, aes(ymin=Performance-sd, ymax=Performance+sd)) +
    geom_point(aes(shape=method), size=2, fill='white') +
    xlab('Number of methods (M)') +
    ylab('Performance (AUC)') +
    theme(legend.position = c(0.75, 0.7)) + scale_color_manual(values=c("FiDEL"="black", "Best_Indv"="grey70")) + scale_shape_manual(values=c("FiDEL"=21, "Best_Indv"=23))+ labs(color  = "Method", shape = "Method") + guides(linetype=FALSE) 


  ggsave(filename, width=6, height=4)
  return (g)
}

cal_partial_performance <- function(.Object, nmethod_list=5:7, nsample=20, seed=100) {
  set.seed(seed)
  x <- c()
  y <- c()
  n <- c()

  for (j in nmethod_list) {
    for (i in 1:nsample) {
      modellist <- sample(1:.Object@nmethods, j)
      FDAuc <- auc_rank(-rowSums(.Object@logit_matrix[,modellist]), .Object@actual_label)
      BestAuc <- max(.Object@actual_performance[modellist])

      x <- c(x, BestAuc)
      y <- c(y, FDAuc)
      n <- c(n, j)
    }
  }

  df <- data.table(FiDEL=y, Best_Indv=x, nmethod=n)
  #df$nmethod <- as.factor(df$nmethod)

  return(df)
}

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

# calculate ensemble score using fermi-dirac statistics
ensemble.fermi <- function(predictions, y, alpha = 1.0, method='+', debug.flag = FALSE) {
  M <- ncol(predictions)
  N <- nrow(predictions)

  res <- matrix(0, nrow=N, ncol=M)
  fd <- matrix(0, nrow=M, ncol=4)
  auclist <- apply(predictions, 2, auc_rank, y)
  fd[, 2:4] <- t(sapply(auclist, get_fermi, attr(y, 'rho'), N=N))
  fd[, 1] <- t(auclist)

  for(m in 1:M) {
    res[ , m] <- frankv(predictions[, m])
    if (method == '+')
      res[ , m] <- -fd[m, 2]^alpha *(fd[m, 4] - res[, m])
    else
      res[ , m] <- 12*N*(fd[m, 1] - 0.5)/(N*N - 1) *((N+1.)/2.- res[, m])
  }

  if(debug.flag) {
    print(fd)
  }

  return(rowSums(res))
}

# find eigen vector of estimate \hat{R} of the rank-one matrix R
# leading eigen vector is close to the true one
# adapted from r-summa (https://github.com/learn-ensemble/R-SUMMA/blob/master/R/matrix_calculations.R)
find_eigen <- function(covMatrix, tol=1e-6, niter_max=10000) {
  eig <- eigen(covMatrix, symmetric=TRUE)
  eig_all <- vector()

  l <- 0
  iter <- 1
  while (abs(l - eig$values[1]) > tol & (iter < niter_max)) {
    l <- eig$values[1]
    r <- (covMatrix - diag(diag(covMatrix)) + diag(l[1]*eig$vectors[, 1]^2))
    eig <- eigen(r, symmetric=TRUE)
    eig_all <- c(eig_all, eig$values[1])
    iter <- iter + 1
  }
  return (list(eig=eig, eig_all=eig_all))
}
