# trainer.R
#
# mtrainer - multi-trainer for caret ML package
#
# author: Sungcheol Kim @ IBM
#
# revision 1.1 - 2020/03/27 - separate trainer and FD ensemble

library(caret)
library(doParallel)
library(tictoc)

#' mtrainer
#'
#' initialize mtrainer object for multiple algorithm container
#'
#' @param x A list of algorithm names (names(caret::getModelInfo()))
#' @param fitControl A list of control variables
#' @return S3 object of list
#' @examples
#' t <- mtrainer(c('C5.0', 'ctree'))
#' @export
new_mtrainer <- function(x = list(), fitControl = NULL) {
  stopifnot(is.character(x))

  # prepare fitControl
  if (is.null(fitControl)) {
    # prepare random seeds
    set.seed(1)
    seeds <- vector(mode = "list", length = 51)
    for(i in 1:50) seeds[[i]] <- sample.int(1000, 22)
    seeds[[51]] <- sample.int(1000, 1)

    fitControl <- caret::trainControl(method = "repeatedcv", repeats = 5,
                                      classProbs = TRUE, summaryFunction = twoClassSummary,
                                      savePredictions = 'final',
                                      search = "random", seeds = seeds)
  }

  structure(list(model_list = x,
                 modelInfo = list(label="FDensemble"),
                 prevalence = numeric(),
                 nmethods = length(x),
                 test_data = list(),
                 test_label = list(),
                 predictions = numeric(),
                 control = fitControl,
                 fitlist = list()),
            class = "mtrainer")
}

# validate
validate_mtrainer <- function(x) {
  modellist_full <- names(caret::getModelInfo())
  check <- x$model_list %in% modellist_full
  if(!all(check)) {
    stop(paste0('Unknown model name: ', x$model_list[!check], '\n'))
  }
  x$nmethods <- length(x$model_list)
  x
}

# helper
mtrainer <- function(x = list(), fitControl = NULL) {
  validate_mtrainer(new_mtrainer(x, fitControl))
}

#' Calculate AUC using rank
#'
#' Function to calculate AUC from score and ground truth label using rank sum formula
#'
#' @param scores A list of values from binary classifier
#' @param y A list of labels
#' @param class1 A name of class 1
#' @return the area under receiver operating curve
#' @examples
#' auc.rank(scores, y)
#' @export
update.mtrainer <- function(mtrainer, newlist) {
  mtrainer$model_list <- newlist
  validate_mtrainer(mtrainer)
}

addmodel.mtrainer <- function(mtrainer, newmodelname) {
  check <- newmodelname %in% mtrainer$model_list
  mtrainer$model_list <- c(mtrainer$model_list, newmodelname[!check])
  validate_mtrainer(mtrainer)
}

train.mtrainer <- function(mtrainer, formula, data, update=FALSE, n_cores=-1) {
  if (n_cores == -1) n_cores <- detectCores() - 1

  # worker module for parallel process
  caret_train <- function(method) {
    message(paste0('Training algorithm : ', method, ' with : ', n_cores, ' cores'))
    flush.console()

    if (method %in% names(mtrainer$fitlist) && !update) {
      message(paste0('... using cached result: ', method))
      mtrainer$fitlist[[method]]
    } else {
      #set.seed(1024)
      if (method %in% c('gbm', 'nnet')) {
        caret::train(formula, data=data, method=method, trControl=mtrainer$control,
                     metric="ROC", tuneLength=4, preProc = c("center", "scale"), verbose=FALSE)
      }
      else {
        caret::train(formula, data=data, method=method, trControl=mtrainer$control,
                     metric="ROC", tuneLength=4, preProc = c("center", "scale"))
      }
    }
  }

  tic(cat('... train model with ', mtrainer$nmethods, ' algorithms\n'))

  cl <- makePSOCKcluster(n_cores)
  registerDoParallel(cl)
  mtrainer$fitlist <- lapply(mtrainer$model_list, caret_train)
  stopCluster(cl)

  names(mtrainer$fitlist) <- mtrainer$model_list
  mtrainer$nmethods <- length(mtrainer$fitlist)
  toc()

  mtrainer
}

predict.mtrainer <- function(mtrainer, newdata=NULL, class1=NULL) {
  message(paste0('... predict using ', mtrainer$nmethods, ' base classifiers'))

  if (is.null(class1)) {
    class1 <- mtrainer$fitlist[[1]]$finalModel$lev[1]
  }
  # check test data set
  if(!is.null(newdata)) mtrainer$test_data <- newdata
  stopifnot(length(mtrainer$test_data) > 0)

  # build predictions
  mtrainer$predictions <- matrix(nrow=nrow(mtrainer$test_data), ncol=mtrainer$nmethods)
  for (i in 1:mtrainer$nmethods) {
    tmp <- predict(mtrainer$fitlist[i], newdata=mtrainer$test_data, type='prob')
    mtrainer$predictions[,i] <- tmp[[1]][, class1]
  }
  colnames(mtrainer$predictions) <- mtrainer$model_list

  mtrainer
}

plot.mtrainer <- function(mtrainer) {
  temp <- resamples(mtrainer$fitlist)
  dotplot(temp)
}

store.mtrainer <- function(mtrainer, filename='temp.RData') {
  cat('... save predictions to ', filename, '\n')
  saveRDS(mtrainer$predictions, file=filename)
}

print.mtrainer <- function(mtrainer, full=TRUE) {
  print(summary.mtrainer(mtrainer, full=full))
}

summary.mtrainer <- function(mtrainer, full=TRUE) {
  # collect informations
}

# make reports over multiple fittings
generate_plot <- function(mtrainerlist, alpha=1, dataname='1', method='mtrainer+', newmodellist=NULL) {
  res <- mtrainerlist %>%
    map(predict, alpha=alpha, method=method, newmodellist=newmodellist) %>%
    map(summary.mtrainer, full=FALSE) %>%
    reduce(rbind)

  medians <- aggregate(ROC ~  model, res, median)

  g <- ggplot(res, aes(x=reorder(model, ROC, FUN = median), ROC, fill=model)) +
    geom_boxplot() +
    geom_text(data=medians, aes(label=formatC(ROC, format = "f", digits = 3), y=ROC-0.02), position = position_dodge2(0.9), size=3) +
    ggtitle(paste0('Data Set ', dataname, ' - 10 iterations - alpha=', alpha)) +
    xlab('Models')

  ggsave(paste0('DS', dataname, '_ROC_10iterations_a', alpha, '_m',
    length(mtrainerlist[[1]]$modelInfo$modelcheck), '.pdf'))
  print(g)
  medians[order(medians$ROC), ]
}

buildmtrainerModel <- function(iter, ml=modellist, df=df, Y=Y, relation=Class~., dataname='1') {
  set.seed(iter)
  inTraining <- createDataPartition(Y, p=.75, list=FALSE)
  training <- df[ inTraining, ]
  testing  <- df[-inTraining, ]
  testingY <- df[-inTraining, ncol(df)]

  temp <- mtrainer(ml) %>%
    train(relation, training, update=FALSE) %>%
    predict(newdata=testing, Y=testingY, method='mtrainer')

  saveRDS(temp, file=paste0("DS", dataname, "_partition", iter, "_mtrainer.RData"))
  temp
}
