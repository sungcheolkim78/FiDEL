# r_summa.R
# SUMMA plus - ensemble method for binary classifier
#
# author: Sungcheol Kim @ IBM

library(caret)
library(purrr)
library(doParallel)
library(tibble)
library(rlist)

# summa S3 object

# initializer
new_summa <- function(x = list(), method = "plus", fitControl = NULL) {
  stopifnot(is.character(x))
  method <- match.arg(method, c("plus", "default"))
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
                 method = "summa",
                 modelInfo = list(label="summa"),
                 control = fitControl,
                 fitlist = list(),
                 testlist = list(),
                 testproblist = list(),
                 roclist = list(),
                 lambdalist = list(),
                 testdata = list(),
                 Y = list(),
                 rho = numeric()),
            method = method,
            class = "summa")
}

# validate
validate_summa <- function(x) {
  modellist_full <- names(caret::getModelInfo())
  check <- x$modellist %in% modellist_full
  if(!all(check)) {
    stop(paste0('unknown model name: ', x$modellist[!check], '\n'))
  }
  x
}

# helper
summa <- function(x = list(), method = "plus", fitControl = NULL) {
  validate_summa(new_summa(x, method, fitControl))
}

# S3 method
update.summa <- function(summa, newlist) {
  summa$modellist <- newlist
  validate_summa(summa)
}

addmodel.summa <- function(summa, newmodelname) {
  check <- newmodelname %in% summa$modellist
  summa$modellist <- c(summa$modellist, newmodelname[!check])
  validate_summa(summa)
}

train.summa <- function(summa, formula, data, update=FALSE, n_cores=-1) {
  if (n_cores == -1) n_cores <- detectCores() - 1

  caret_train <- function(method) {
    message(paste0('Training algorithm : ', method, ' with : ', n_cores, ' cores'))
    flush.console()

    if (method %in% names(summa$fitlist) && !update) {
      message(paste0('... using cached result: ', method))
      summa$fitlist[[method]]
    } else {
      #set.seed(1024)
      if (method %in% c('gbm', 'nnet'))
        caret::train(formula, data=data, method=method, trControl=summa$control,
                     metric="ROC", tuneLength=4, preProc = c("center", "scale"), verbose=FALSE)
      else
        caret::train(formula, data=data, method=method, trControl=summa$control,
                     metric="ROC", tuneLength=4, preProc = c("center", "scale"))
    }
  }

  cl <- doParallel::makePSOCKcluster(n_cores)
  doParallel::registerDoParallel(cl)
  summa$fitlist <- purrr::map(summa$modellist, caret_train)
  doParallel::stopCluster(cl)

  names(summa$fitlist) <- summa$modellist
  summa
}

predict.summa <- function(summa, newdata = NULL, Y=NULL, alpha=1.0, newmodellist = NULL,
                          method='summa+') {
  msg <- paste0('... predict using ', method, ' , alpha: ', alpha)
  message(msg)

  # check test data set
  if(!is.null(newdata)) summa$testdata <- newdata
  stopifnot(length(summa$testdata) > 0)

  # check test class data set
  if(!is.null(Y)) summa$Y <- Y
  stopifnot(length(summa$Y) > 0)

  # check models for training
  if(is.null(newmodellist)) newmodellist <- names(summa$fitlist)

  if(length(summa$testlist) > 0) {
    check <- newmodellist %in% names(summa$testlist)
    newmodellist <- newmodellist[check]
    if (!all(newmodellist %in% summa$modellist))
      stop("Add model and train first")
  } else {
    message(paste0('... predict using initial ', length(summa$fitlist), ' classifiers'))
    check <- newmodellist == ' '
    summa$testlist <- purrr::map(summa$fitlist, predict, newdata=summa$testdata)
    summa$testproblist <- purrr::map(summa$fitlist, predict, newdata=summa$testdata, type='prob')
  }

  # Y should be factor
  class1name = levels(summa$Y)[[1]]
  class2name = levels(summa$Y)[[2]]
  summa$rho <- sum(summa$Y == class1name)/length(summa$Y)

  # create probability matrix
  if (any(!check)) {
    message(paste0('... predict using additional ', sum(!check), ' classifiers'))
    list.append(summa$testlist, map(summa$fitlist[!check], predict, newdata=summa$testdata))
    list.append(summa$testproblist, map(summa$fitlist[!check], predict, newdata=summa$testdata, type='prob'))
  }

  # create roc, labmda, rstar list
  summa$roclist <- purrr::map(summa$testproblist, rocrank, reference = summa$Y)
  summa$lambdalist <- purrr::map(summa$roclist, lambda_fromROC, N=length(summa$Y), rho=summa$rho)

  # calculate new score using SUMMA algorithm
  if (method == 'summa')
    res <- cal_score_summa(summa, newmodellist = newmodellist)
  else
    res <- cal_score_summap(summa, alpha = alpha, newmodellist = newmodellist)
  if (alpha > 1) method <- paste0('summa+', alpha)

  # add results
  summa$testlist[[method]] <- as.factor(ifelse(res > 0, class2name, class1name))
  summa$testproblist[[method]] <- data.frame(1/(1 + exp(res)), 1/(1+exp(-res)))
  names(summa$testproblist[[method]]) <- c(class1name, class2name)

  summa$roclist[method] <- rocrank(summa$testproblist[[method]], reference = summa$Y)
  summa$lambdalist[[method]] <- lambda_fromROC(summa$roclist[[method]], N=length(summa$Y), rho=summa$rho)
  summa$confmatrix <- purrr::map(summa$testlist, confusionMatrix, reference = summa$Y)
  summa$modelInfo$modelcheck <- names(summa$roclist) %in% c(newmodellist, method)

  summa
}

plot.summa <- function(summa) {
  temp <- caret::resamples(summa$fitlist)
  dotplot(temp)
}

print.summa <- function(summa, full=TRUE) {
  print(summary.summa(summa, full=full))
}

summary.summa <- function(summa, full=TRUE) {
  # collect informations
  ROC <- unlist(summa$roclist)
  l1 <- purrr::map_dbl(summa$lambdalist, 1)
  l2 <- purrr::map_dbl(summa$lambdalist, 2)
  rs <- purrr::map_dbl(summa$lambdalist, 3)

  # prepare data.frame
  res <- tibble::tibble(ROC=ROC, l1=l1, l2=l2, rs=rs, model=names(summa$roclist),
                chk=summa$modelInfo$modelcheck)
  res <- res[order(res$ROC), ]

  # return results
  if (full) return(res)
  res[res$chk, 1:5]
}

# Utility functions

cal_score_summap <- function(summa, newmodellist = NULL, alpha = 1.0, view = FALSE) {
  if (is.null(newmodellist)) newmodellist <- summa$modellist

  class1name = levels(summa$Y)[[1]]

  res <- matrix(0, nrow=length(summa$testlist[[1]]), ncol=length(newmodellist))
  colnames(res) <- newmodellist

  for(m in newmodellist) {
    res[ , m] <- rank(summa$testproblist[[m]][[class1name]])
    res[ , m] <- summa$lambdalist[[m]][[2]]^alpha *(summa$lambdalist[[m]][[3]] - res[, m])
  }

  if(view) {
    temp <- as.data.frame(res)
    temp[['summa+']] <- rowMeans(res)
    names(temp)[-ncol(temp)] <- c(summa$modellist)
    plot(temp)
    print(cor(temp, method = "spearman"))
  }

  rowMeans(res)
}

cal_score_summa <- function(summa, newmodellist = NULL, view = FALSE) {
  if (is.null(newmodellist)) newmodellist <- summa$modellist

  class1name = levels(summa$Y)[[1]]
  N = length(summa$Y)

  res <- matrix(0, nrow=length(summa$testlist[[1]]), ncol=length(newmodellist))
  colnames(res) <- newmodellist

  for(m in newmodellist) {
    res[ , m] <- rank(summa$testproblist[[m]][[class1name]])
    res[ , m] <- 12*N*(summa$roclist[[m]] - 0.5)/(N*N - 1) *((N+1.)/2.- res[, m])
  }

  if(view) {
    temp <- as.data.frame(res)
    temp$summa <- rowMeans(res)
    names(temp)[-ncol(temp)] <- c(summa$modellist)
    plot(temp)
    print(cor(temp, method = "spearman"))
  }

  rowMeans(res)
}

# calculate ROC from rank and reference
rocrank <- function(problist, reference) {
  class1name = levels(reference)[[1]]
  class2name = levels(reference)[[2]]

  temp <- data.frame(prob=problist[[class1name]], truth=reference)
  temp <- temp[order(temp$prob, decreasing = TRUE), ]
  temp$rank <- seq_along(reference)

  (mean(temp$rank[temp$truth == class2name]) - mean(temp$rank[temp$truth == class1name]))/length(reference) + 0.5
}

# calculate lambda1,2 from ROC, rho
lambda_fromROC <- function(ROC, N=N, rho=rho) {
  costFunc <- function(l, rho, auroc, N) {
    r <- 1:N
    sum1 <- sum(1/(1+exp(l[2]*r - l[1])))/N
    sum2 <- sum(r/(1+exp(l[2]*r - l[1])))/(N*N*rho)

    (rho - sum1)^2 + (1 + .5/N - rho/2 - auroc*(1-rho) - sum2)^2
  }
  temp <- optim(c(N*rho*0.1, 0.1), costFunc, rho=rho, auroc=ROC, N=N)
  l1 <- -temp$par[1]
  l2 <- temp$par[2]
  rs <- 1/l2 * log((1 - rho)/rho) - l1/l2
  return(c(l1 = l1, l2 = l2, rs = rs))
}

# calculate lambda1, lambda2 from ROC, rho (version 2)
lambda_fromROC_appr <- function(ROC, N=N, rho=rho) {
  l1_low <- log(1/rho - 1) - 12*N*(ROC-0.5)/(N*N-1)*((N+1+N*rho)*0.5 - N*rho*ROC)
  l2_low <- 12*N*(ROC-0.5)/(N*N-1)

  temp <- sqrt(rho*(1-rho)*(1-2*(ROC-0.5)))
  l1_high <- -2*rho/(sqrt(3)*temp)
  l2_high <- 2/(sqrt(3)*N*temp)

  alpha <- 2*(ROC - 0.5)
  l1 <- l1_high*alpha + l1_low*(1-alpha)
  l2 <- l2_high*alpha + l2_low*(1-alpha)
  return(c(l1low=l1_low, l2low=l2_low, l1high=l1_high, l2high=l2_high, l1=l1, l2=l2))
}

# make reports over multiple fittings
generate_plot <- function(summalist, alpha=1, dataname='1', method='summa+', newmodellist=NULL) {
  res <- summalist %>%
    purrr::map(predict, alpha=alpha, method=method, newmodellist=newmodellist) %>%
    purrr::map(summary.summa, full=FALSE) %>%
    purrr::reduce(rbind)

  medians <- aggregate(ROC ~  model, res, median)

  g <- ggplot(res, aes(x=reorder(model, ROC, FUN = median), ROC, fill=model)) +
    geom_boxplot() +
    geom_text(data=medians, aes(label=formatC(ROC, format = "f", digits = 3), y=ROC-0.02), position = position_dodge2(0.9), size=3) +
    ggtitle(paste0('Data Set ', dataname, ' - 10 iterations - alpha=', alpha)) +
    xlab('Models')

  ggsave(paste0('DS', dataname, '_ROC_10iterations_a', alpha, '_m',
    length(summalist[[1]]$modelInfo$modelcheck), '.pdf'))
  print(g)
  medians[order(medians$ROC), ]
}

buildSummaModel <- function(iter, ml=modellist, df=df, Y=Y, relation=Class~., dataname='1') {
  set.seed(iter)
  inTraining <- caret::createDataPartition(Y, p=.75, list=FALSE)
  training <- df[ inTraining, ]
  testing  <- df[-inTraining, ]
  testingY <- df[-inTraining, ncol(df)]

  temp <- summa(modellist) %>%
    train(relation, training, update=FALSE) %>%
    predict(newdata=testing, Y=testingY, method='summa')

  saveRDS(temp, file=paste0("DS", dataname, "_partition", iter, "_summa.RData"))
  temp
}
