#methods for training each algorithm in the same partition
train.mtrainer.slm.same.partition <- function(mtrainer, formula, data_list, update=FALSE, n_cores=-1) {
  if (n_cores == -1) n_cores <- detectCores() - 1

  fname <- paste0(mtrainer$dataInfo, 'samepartition.RData')

  if (file.exists(fname) & !update) { mtrainer$fitlist <- readRDS(fname) }

  tic(cat('... train model with ', mtrainer$nmethods, ' algorithms\n'))

  for (i in 1:mtrainer$nmethods) {

  if (mtrainer$model_list[i] %in% names(mtrainer$fitlist) && !update) {
      message(paste0('... using cached result: ', mtrainer$model_list[i]))
  } else {

      if (length(data_list) == 1) {
          fit <- caret_train_slm(mtrainer$model_list[i], mtrainer, formula, data_list[[1]], n_cores)
      } else {
          fit <- caret_train_slm(mtrainer$model_list[i], mtrainer, formula, data_list[[1]], n_cores)
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

train.mtrainer.same.partition <- function(mtrainer, formula, data_list, update=FALSE, n_cores=-1) {
  if (n_cores == -1) n_cores <- detectCores() - 1
  fname <- paste0(mtrainer$dataInfo, 'samepartition2.RData')
  if (file.exists(fname) & !update) { mtrainer$fitlist <- readRDS(fname) }

  tic(cat('... train model with ', mtrainer$nmethods, ' algorithms\n'))

  # train multiple data with different methods
  for (i in 1:mtrainer$nmethods) {
    # check fit data
    if (mtrainer$model_list[i] %in% names(mtrainer$fitlist) && !update) {
      message(paste0('... using cached result: ', mtrainer$model_list[i]))
    } else {
      # single training data case
      if (length(data_list) == 1)
        fit <- caret_train(mtrainer$model_list[i], mtrainer, formula, data_list[[1]], n_cores)
      else
        fit <- caret_train(mtrainer$model_list[i], mtrainer, formula, data_list[[5]], n_cores)

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

auc_rank_train_same_partition <- function(scores, y,  nmethods, x, class1=NULL) {



  exclude.folds <- setdiff(1:22, 2:nmethods)
  y <- y[-exclude.folds]
  y <- do.call(rbind, y) %>% as_tibble() %>% pull(y)

  nan.values <- which(is.na(scores))

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
  mat$rank <- frankv(scores, order=-1)

  res <- abs(sum(mat$rank[y == attr(y, 'class1')])/N1 - sum(mat$rank[y == attr(y, 'class2')])/N2)/N + 0.5

  if (res < 0.5) {
    message('... class label might be wrong.')
    res <- 1 - 0.5
  }

  return (res)
}


predict.mtrainer.train.same.partition <- function(mtrainer, newdata2=NULL, class1=NULL) {

  message(paste0('... predict using ', mtrainer$nmethods, ' base classifiers'))

  if (is.null(class1)) {
    class1 <- mtrainer$fitlist[[1]]$finalModel$obsLevels[1]
  }

  exclude.folds <- setdiff(1:22, 2:mtrainer$nmethods)
  newdata2 <- newdata2[-exclude.folds]

  nrows <- unlist(lapply(1:length(newdata2), function(x) newdata2[[x]] %>% as_tibble() %>% nrow()))

  tot.nrow <- sum(nrows)

  mtrainer$predictions <- matrix(nrow=tot.nrow, ncol=mtrainer$nmethods)

  newdata <- do.call(rbind, newdata2) %>% as_tibble()

  for (j in 1:mtrainer$nmethods){

      if(!is.null(newdata)) {
        mtrainer$test_data <- newdata
      }

      stopifnot(!is.null(mtrainer$test_data))

      tmp <- predict(mtrainer$fitlist[j], newdata=mtrainer$test_data, type='prob')

      pred <- tmp[[1]][, class1]

      mtrainer$predictions[, j] <- pred

  }

  colnames(mtrainer$predictions) <- mtrainer$model_list
  
  return(mtrainer)
}