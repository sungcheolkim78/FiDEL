#computes AUC in the training set excluding the partition that the base method was learned on
auc_rank_train <- function(scores, y,  nmethods, x, class1=NULL) {


  nrows <- unlist(lapply(1:nmethods, function(x) y[[x]] %>% as_tibble() %>% nrow()))

  exclude.folds <- setdiff(1:22, 1:nmethods)
  y <- y[-exclude.folds]
  y <- do.call(rbind, y) %>% as_tibble() %>% pull(y)

  cumsum.nrows <- c(0, cumsum(nrows))

  remove.idx <- (cumsum.nrows[x] + 1): cumsum.nrows[x+1]

  scores <- scores[-remove.idx]

  nan.values <- which(is.na(scores))

  y <- y[-remove.idx]

  if (!identical(integer(0), nan.values)){

    y <- y[-nan.values]
    scores <- scores[-nan.values]

  }


  # validate inputs
  stopifnot(length(scores) == length(y))
  if (is.null(attr(y, 'rho')) || attr(y, 'rho') == 0) { y <- to_label(y, class1=class1) }

  # calculate class 1 and class 2
  N <- attr(y, 'N')
  N1 <- attr(y, 'N1')
  N2 <- attr(y, 'N2')
  mat <- data.table(scores=scores, y=y)
  mat$rank <- frankv(scores, order=-1, ties.method="random")

  res <- abs(sum(mat$rank[y == attr(y, 'class1')])/N1 - sum(mat$rank[y == attr(y, 'class2')])/N2)/N + 0.5

  if (res < 0.5) {
    message('... class label might be wrong.')
    res <- 1 - 0.5
  }

  return (res)
}

not_any_na <- function(x) all(!is.na(x))

#predicts the class label usnig the training set estimated FiDEL parameters
predict.mtrainer.train <- function(mtrainer, newdata2=NULL, class1=NULL) {

  message(paste0('... predict using ', mtrainer$nmethods, ' base classifiers'))

  if (is.null(class1)) {
    class1 <- mtrainer$fitlist[[1]]$finalModel$obsLevels[1]
  }

  exclude.folds <- setdiff(1:22, 1:mtrainer$nmethods)
  newdata2 <- newdata2[-exclude.folds]

  nrows <- unlist(lapply(1:mtrainer$nmethods, function(x) newdata2[[x]] %>% as_tibble() %>% nrow()))

  cumsum.nrows <- c(0, cumsum(nrows))

  tot.nrow <- sum(nrows)

  mtrainer$predictions <- matrix(nrow=tot.nrow, ncol=mtrainer$nmethods)

  newdata3 <- do.call(rbind, newdata2) %>% as_tibble()

  for (j in 1:mtrainer$nmethods){

      remove.idx <- (cumsum.nrows[j] + 1): cumsum.nrows[j+1]

      newdata <- newdata3[-remove.idx,]

      if(!is.null(newdata)) {
        mtrainer$test_data <- newdata
      }

      stopifnot(!is.null(mtrainer$test_data))

      tmp <- predict(mtrainer$fitlist[j], newdata=mtrainer$test_data, type='prob')

      pred <- tmp[[1]][, class1]

      mtrainer$predictions[-remove.idx, j] <- pred

  }

  colnames(mtrainer$predictions) <- mtrainer$model_list
  
  return(mtrainer)
}

#plots the overall_performance comparing the FiDEL, WoC, and Best Individual classifier
overall_performance <- function(fde.obj, woc.obj, nmethod_list, nsample, seed, method){

  cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

  fde5 <- fde.obj
  woc5 <- woc.obj

  df <- cal_partial_performance(fde5, nmethod_list=nmethod_list, nsample=nsample, seed=seed)

  df2 <- cal_partial_performance(woc5, nmethod_list=nmethod_list, nsample=nsample, seed=seed)

  df$WoC <- df2$FiDEL

  df <- melt(df, id.vars = 'nmethod', variable.name='method', value.name='AUC')

  tmp <- df %>% group_by(nmethod, method) %>%
      mutate(Performance=mean(AUC)) %>%
      mutate(sd=sd(AUC)) %>%
      mutate(N=length(AUC))

    if (method == 'SE') {
      tmp$sd <- tmp$sd/sqrt(tmp$N)
    }

    tmp <- tmp %>% mutate(shape=ifelse(method=="FiDEL", "21", ifelse(method=="WoC", "25", "23")))
    #c(0.75, 0.7)
    g <- ggplot(tmp, aes(x=nmethod, y=Performance)) + theme_classic() +
      geom_line(aes(linetype=method, color=method), size=2) +
      geom_errorbar(width=.1, aes(ymin=Performance-sd, ymax=Performance+sd)) +
      geom_point(aes(shape=method), size=2, fill='white') +
      xlab('Number of methods (M)') +
      ylab('Performance (AUC)') +
      theme(legend.position = c(0.75, 0.17)) + scale_color_manual(values=c("FiDEL"=cbPalette[3], "WoC"="black", "Best_Indv"="grey70")) + scale_shape_manual(values=c("FiDEL"=21, "Best_Indv"=23, "WoC"=25))+ labs(color  = "Method", shape = "Method") + guides(linetype=FALSE) 

}

#takes in FiDEL object and produces correlation matrix of ranks of test set items from base classifiers
corrank <- function(object, type="weight"){

  rankmatrix <- object@rank_matrix
  labels <- as.character(object@actual_label)
  labels <- ifelse(labels=="Yes", 1, 0)

  class1freq <- mean(labels)

  ranker <- rankmatrix %>% as_tibble() %>% mutate(labels=labels)

  #C5.0 has all equal ranks, break ranks randomly

  ranker$C5.0 <- sample(1:dim(ranker)[1], replace=FALSE, size=dim(ranker)[1])
  
  split.data <- split(ranker, ranker$labels)


  mats <- lapply(split.data, function(x) split.data$"0" %>% select(-c(labels, C5.0, A_FD)) %>% cor() %>% round(2) )

  if (type=="weight"){
    cor_m <- mats$"0"*(1-class1freq) + mats$"1"*class1freq
  } 

  if (type=="equal_weight"){
    cor_m <- (mats$"0" + mats$"1")/2
  }

  #cor_m[upper.tri(cor_m)] <- NA

  #melted_cor_m <- reshape2::melt(cor_m, na.rm=TRUE)

  #melted_cor_m

  cor_m
}

#Computes the empirical probability of class given rank vector for each of the base methods 
#note: we can compare the output of fidel.fits() with fd.coords() to understand how well the empirical and analytical probability of class given rank vector are
fidel.fits <- function(t1, traininglist, n.iter=200, n=1000){

  mtrainer <- t1
  newdata2 <- traininglist


  pcr.matrix <- matrix(ncol=mtrainer$nmethods, nrow=n)

  for (i in 1:mtrainer$nmethods){

    exclude.folds <- setdiff(1:22, 1:mtrainer$nmethods)
    newdata2 <- newdata2[-exclude.folds]
    newdata2 <- newdata2[-i]
    newdata3 <- do.call(rbind, newdata2) %>% as_tibble()

    rho <- do.call(rbind, traininglist) %>% as_tibble() %>% mutate(p=ifelse(as.character(y)=="Yes", 1, 0)) %>% pull(p) %>% mean()

    split.data <- split(newdata3, newdata3$y)
    yes.data <- split.data$Yes
    no.data <- split.data$No

    pcr <- vector("numeric", n)

    nclass1 <- round(n*rho)
    nclass0 <- n - nclass1

    for (j in 1:n.iter){

      idx.rows.class0 <- sample(x = dim(no.data)[1], size = nclass0, replace = FALSE)
      idx.rows.class1 <- sample(x = dim(yes.data)[1], size = nclass1, replace = FALSE)

      data <- rbind(no.data[idx.rows.class0, ], yes.data[idx.rows.class1, ]) %>% as_tibble()
      true.labels <- data$y

      tmp <- predict(mtrainer$fitlist[i], newdata=data, type='prob')
      #maybe add ties.method="random"
      pred <- 1 + n - (tmp[[1]][, "Yes"] %>% frankv(ties.method="random"))

      class_ranked <- data.frame(pred=pred, true=true.labels) %>% as_tibble() %>% arrange(pred) %>% mutate(t=ifelse(as.character(true)=="Yes", 1, 0)) %>% pull(t)

      pcr <- pcr + class_ranked
    }

    pcr <- pcr / n.iter

    pcr.matrix[,i] <- pcr

  }
}

#defines fermi-dirac distribution
fd <- function(x, beta, mu){
  r <- c(1:x)
  
  1 / (1 + exp(beta * (r - mu)))
}

#computes the fermi-dirac distribution for each of the base classifiers given the beta and mu values estimated from the training set
fd.coords <- function(fde4, pcr.matrix){

  lapply(1:length(fde4@beta), function(i) fd(x=dim(pcr.matrix)[1], beta=fde4@beta[i], mu=fde4@mu[i]))

}


#helper functions for training base classifiers for the springleaf marketing dataset
caret_train_slm <- function(method, mtrainer, formula, tr_data, n_cores) {
  message(paste0('Training algorithm : ', method, ' with : ', n_cores, ' cores'))
  flush.console()

  cl <- makePSOCKcluster(n_cores, setup_strategy="sequential")
  registerDoParallel(cl)
  if (method %in% c('gbm', 'nnet')) {

      if (method=="nnet"){
        fit <- caret::train(y~., data=tr_data, method='nnet', trControl=mtrainer$control, metric="ROC", tuneLength=4, preProc = c("center", "scale"), verbose=FALSE, threshold=0.3)
      } else{
        fit <- caret::train(formula, data=tr_data, method=method, trControl=mtrainer$control, metric="ROC", tuneLength=4, preProc = c("center", "scale"), verbose=FALSE, threshold=0.3)
      }

  } else {
    fit <- caret::train(formula, data=tr_data, method=method, trControl=mtrainer$control, metric="ROC", tuneLength=4, preProc = c("center", "scale"))
  }
  stopCluster(cl)
  return (fit)
}

train.mtrainer.slm <- function(mtrainer, formula, data_list, update=FALSE, n_cores=-1) {
  if (n_cores == -1) n_cores <- detectCores() - 1

  fname <- paste0(mtrainer$dataInfo, '.RData')

  if (file.exists(fname) & !update) { mtrainer$fitlist <- readRDS(fname) }

  tic(cat('... train model with ', mtrainer$nmethods, ' algorithms\n'))

  for (i in 1:mtrainer$nmethods) {

  if (mtrainer$model_list[i] %in% names(mtrainer$fitlist) && !update) {
      message(paste0('... using cached result: ', mtrainer$model_list[i]))
  } else {

      if (length(data_list) == 1) {
          fit <- caret_train_slm(mtrainer$model_list[i], mtrainer, formula, data_list[[1]], n_cores)
      } else {
          fit <- caret_train_slm(mtrainer$model_list[i], mtrainer, formula, data_list[[i]], n_cores)
      }
      fitlist <- list(fit)
      names(fitlist) <- c(mtrainer$model_list[i])
      mtrainer$fitlist <- append(mtrainer$fitlist, fitlist)
    }
    saveRDS(mtrainer$fitlist, file = fname)
  }

  mtrainer$nmethods <- length(mtrainer$fitlist)
  toc()
  mtrainer
}



