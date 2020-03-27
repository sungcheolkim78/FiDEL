# trainer.R
#
# mtrainer - multi-trainer for caret ML package
#
# author: Sungcheol Kim @ IBM
#
# revision 1.1 - 2020/03/27 - separate trainer and FD ensemble

library(caret)
library(purrr)
library(doParallel)
library(tibble)
library(rlist)

# mtrainer S3 object

# initializer
new_mtrainer <- function(x = list(), method = "plus", fitControl = NULL) {
  stopifnot(is.character(x))
  method <- match.arg(method, c("plus", "default"))

  # prepare fitControl
  if (is.null(fitControl)) {
    set.seed(1)
    seeds <- vector(mode = "list", length = 51)
    for(i in 1:50) seeds[[i]] <- sample.int(1000, 22)
    seeds[[51]] <- sample.int(1000, 1)

    fitControl <- trainControl(method = "repeatedcv", repeats = 5,
                               classProbs = TRUE, summaryFunction = twoClassSummary,
                               search = "random", seeds = seeds)
  }

  structure(list(modellist = x,
                 method = "mtrainer",
                 modelInfo = list(label="mtrainer"),
                 control = fitControl,
                 fitlist = list(),
                 testlist = list(),
                 predictions = list(),
                 performances = list(),
                 testdata = list(),
                 actual_label = list(),
                 rho = numeric()),
            method = method,
            class = "mtrainer")
}

# validate
validate_mtrainer <- function(x) {
  modellist_full <- names(caret::getModelInfo())
  check <- x$modellist %in% modellist_full
  if(!all(check)) {
    stop(paste0('unknown model name: ', x$modellist[!check], '\n'))
  }
  x
}

# helper
mtrainer <- function(x = list(), method = "plus", fitControl = NULL) {
  validate_mtrainer(new_mtrainer(x, method, fitControl))
}

# S3 method
update.mtrainer <- function(mtrainer, newlist) {
  mtrainer$modellist <- newlist
  validate_mtrainer(mtrainer)
}

addmodel.mtrainer <- function(mtrainer, newmodelname) {
  check <- newmodelname %in% mtrainer$modellist
  mtrainer$modellist <- c(mtrainer$modellist, newmodelname[!check])
  validate_mtrainer(mtrainer)
}

train.mtrainer <- function(mtrainer, formula, data, update=FALSE, n_cores=-1) {
  if (n_cores == -1) n_cores <- detectCores() - 1

  caret_train <- function(method) {
    message(paste0('Training algorithm : ', method, ' with : ', n_cores, ' cores'))
    flush.console()

    if (method %in% names(mtrainer$fitlist) && !update) {
      message(paste0('... using cached result: ', method))
      mtrainer$fitlist[[method]]
    } else {
      #set.seed(1024)
      if (method %in% c('gbm', 'nnet'))
        caret::train(formula, data=data, method=method, trControl=mtrainer$control,
                     metric="ROC", tuneLength=4, preProc = c("center", "scale"), verbose=FALSE)
      else
        caret::train(formula, data=data, method=method, trControl=mtrainer$control,
                     metric="ROC", tuneLength=4, preProc = c("center", "scale"))
    }
  }

  cl <- makePSOCKcluster(n_cores)
  registerDoParallel(cl)
  mtrainer$fitlist <- map(mtrainer$modellist, caret_train)
  stopCluster(cl)

  names(mtrainer$fitlist) <- mtrainer$modellist
  mtrainer
}

predict.mtrainer <- function(mtrainer, newdata = NULL, Y=NULL, alpha=1.0, newmodellist = NULL,
                          method='mtrainer+') {
  msg <- paste0('... predict using ', method, ' , alpha: ', alpha)
  message(msg)

  # check test data set
  if(!is.null(newdata)) mtrainer$testdata <- newdata
  stopifnot(length(mtrainer$testdata) > 0)

  # check test class data set
  if(!is.null(Y)) mtrainer$Y <- as_label(Y)
  stopifnot(length(mtrainer$Y) > 0)

  # check models for training
  if(is.null(newmodellist)) newmodellist <- names(mtrainer$fitlist)

  if(length(mtrainer$testlist) > 0) {
    check <- newmodellist %in% names(mtrainer$testlist)
    newmodellist <- newmodellist[check]
    if (!all(newmodellist %in% mtrainer$modellist))
      stop("Add model and train first")
  } else {
    message(paste0('... predict using initial ', length(mtrainer$fitlist), ' classifiers'))
    check <- newmodellist == ' '
    mtrainer$testlist <- map(mtrainer$fitlist, predict, newdata=mtrainer$testdata)
    mtrainer$predictions <- apply(mtrainer$fitlist, predict, newdata=mtrainer$testdata, type='prob')[[1]]
  }
  print(mtrainer$predictions)

  # Y should be factor
  mtrainer$rho <- attr(Y, 'rho')

  # create probability matrix
  if (any(!check)) {
    message(paste0('... predict using additional ', sum(!check), ' classifiers'))
    list.append(mtrainer$testlist, map(mtrainer$fitlist[!check], predict, newdata=mtrainer$testdata))
    list.append(mtrainer$testproblist, map(mtrainer$fitlist[!check], predict, newdata=mtrainer$testdata, type='prob'))
  }

  # create roc, labmda, rstar list
  mtrainer$roclist <- map(mtrainer$testproblist, auc.rank, mtrainer$Y)

  # calculate new score using mtrainer algorithm
  fde1 <- fde(mtrainer$testproblist)
  fde1 <- predict_performance(fde1, mtrainer$auclist, mtrainer$rho)

  return (fde1)
}

plot.mtrainer <- function(mtrainer) {
  temp <- resamples(mtrainer$fitlist)
  dotplot(temp)
}

print.mtrainer <- function(mtrainer, full=TRUE) {
  print(summary.mtrainer(mtrainer, full=full))
}

summary.mtrainer <- function(mtrainer, full=TRUE) {
  # collect informations
  ROC <- unlist(mtrainer$roclist)
  l1 <- map_dbl(mtrainer$lambdalist, 1)
  l2 <- map_dbl(mtrainer$lambdalist, 2)
  rs <- map_dbl(mtrainer$lambdalist, 3)

  # prepare data.frame
  res <- tibble(ROC=ROC, l1=l1, l2=l2, rs=rs, model=names(mtrainer$roclist),
                chk=mtrainer$modelInfo$modelcheck)
  res <- res[order(res$ROC), ]

  # return results
  if (full) return(res)
  res[res$chk, 1:5]
}

# make reports over multiple fittings
generate_plot <- function(mtrainerlist, alpha=1, dataname='1', method='mtrainer+', newmodellist=NULL) {
  res <- mtrainerlist %>%
    map(predict, alpha=alpha, method=method, newmodellist=newmodellist) %>%
    map(mtrainerry.mtrainer, full=FALSE) %>%
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
