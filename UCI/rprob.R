# Utility functions
#
# Author: Sung-Cheol Kim @ IBM
# Date: 2019/12/12 (initial)

# load libraries
library(doParallel)
library(caret)
library(tictoc)
library(precrec)
library(ROCit)
library(tidyr)
library(ggplot2)

# training model with multicore
multicore_train <- function(modelName, relationship, trainingD, fitControl, tuneGrid, n_cores=6, dataName='data', save=FALSE) {
  # set clock and multicore
  tic(sprintf('Train Time (%s):', modelName))
  cl <- makePSOCKcluster(n_cores)
  registerDoParallel(cl)
  
  # training 
  Fit <- switch(modelName, 
                "gbm" = train(relationship, data = trainingD, method = "gbm", trControl = fitControl, verbose = FALSE, 
                              tuneGrid = gbmGrid, metric = "ROC"),
                "svm" = train(relationship, data = trainingD, method = "svmRadial", trControl = fitControl, 
                              preProc = c("center", "scale"), tuneLength = tuneGrid, metric = "ROC"), 
                "rda" = train(relationship, data = trainingD, method = "rda", trControl = fitControl, 
                              tuneLength = tuneGrid, metric = "ROC"),
                "pls" = train(relationship, data = trainingD, method = "pls", trControl = fitControl, 
                              preProc = c("center", "scale"), tuneLength = tuneGrid, metric = "ROC"), 
                "lgb" = train(relationship, data = trainingD, method = "LogitBoost", trControl = fitControl, 
                              preProc = c("center", "scale"), tuneLength = tuneGrid, metric = "ROC"),
                "pRF" = train(relationship, data = trainingD, method = "parRF", trControl = fitControl, 
                              preProc = c("center", "scale"), tuneLength = tuneGrid, metric = "ROC"),
                "adb" = train(relationship, data = trainingD, method = "adaboost", trControl = fitControl, 
                              preProc = c("center", "scale"), tuneLength = tuneGrid, metric = "ROC"),
                "avN" = train(relationship, data = trainingD, method = "avNNet", trControl = fitControl, 
                              preProc = c("center", "scale"), tuneLength = tuneGrid, metric = "ROC")

		)

  stopCluster(cl)
  toc()

  attr(Fit, "dataName") <- dataName
  print(Fit$bestTune)

  if (save) save(Fit, file=paste0(dataName, '_', modelName, '.RData'))

  return(Fit)
}

# create AUROC 
modelplot.roc <- function(Fit, df, classes, rho, with.cm=FALSE, save.pdf=FALSE) {
  # calculate probability for newdata
  if ("summaplus" %in% class(Fit)) {
    modelName <- attr(Fit, "modelName")
    con.data <- predict(Fit, newdata = df, rho = rho, type='response')
    print(head(con.data))
  }
  else {
    modelName <- Fit$modelInfo$label
    con.data <- predict(Fit, newdata = df)
    print(head(con.data))
  }
  if (with.cm) print(confusionMatrix(data = con.data, reference = classes))
  
  temp <- predict(Fit, newdata = df, rho=rho, type='prob')[[attr(rho, "Positive")]]
  fit.roc <- rocit(score = temp, class = as.character(classes), negref = attr(rho, "Negative"))
  
  #dev.new(width=6, height=6)
  plot(fit.roc)
  title(main = modelName)
  text(0.5, 0.5, sprintf("AUC = %.4f", fit.roc$AUC), adj=c(0,0))

	if (save.pdf) {
    filename <- paste0("ROC-", attr(Fit, "dataName"), "-", modelName, ".pdf")
    dev.copy(pdf, file=filename, width=6, height=6)
    cat(sprintf("... save to %s", filename))
    dev.off()
  }

  return(fit.roc)
}

# create rank probability for each fit model
modelplot.rank <- function(Fits, df, rho) {
  res <- data.frame(rank = 1:length(df[[1]]))
  colname_array <- 'rank'
  
  for (Fit in Fits) {
    temp <- predict(Fit, newdata=df, rho=rho, type='prob')[[attr(rho, "Positive")]]
    if ("summaplus" %in% class(Fit)) modelName <- attr(Fit, "modelName")
    else modelName <- Fit$modelInfo$label

    temp <- sort(temp, decreasing = TRUE)
    res <- cbind(res, temp)
    colname_array <- c(colname_array, modelName)
  }
  
  colnames(res) <- colname_array

  res <- res %>%
    gather(colname_array[-1], key='method', value = 'prob')
  
  g <- ggplot(res, aes(rank, prob)) + 
       geom_point(aes(color=method)) + 
       geom_line(aes(color=method)) + 
       theme(legend.position = 'bottom') +
       guides(color=guide_legend(nrow=2)) +
       xlab('Rank') + ylab('P(1|r)')
  
  return(g)
}

# calculate rank probability by bootstrap method 
iterate.model <- function(modelName, df, classes, rho, relationship, iterNumber, tuneGrid, n_cores=6) {
  fitControl <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 10,
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary)
  
  # prepare output array
  res <- data.frame(l1=numeric(0), l2=numeric(0), rs=numeric(0), method=character(0))
  
  # start clock
  tic(sprintf("Compute: (%s) %i iteration", modelName, iterNumber))
  cl <- makePSOCKcluster(n_cores)
  registerDoParallel(cl)
  
	# calculate for multiple partitions
  for(i in 1:iterNumber) {
    inTraining.temp <- caret::createDataPartition(classes, p = .75, list = FALSE)
    training.temp <- df[ inTraining.temp,]
    testing.temp  <- df[-inTraining.temp,]
    testing.tempY <- classes[-inTraining]
    
    Fit <- switch(modelName, 
                  "gbm" = caret::train(relationship, data = training.temp, method = "gbm", 
                                       trControl = fitControl, tuneGrid = tuneGrid,
                                       verbose = F, metric = "ROC"), 
                  "svm" = caret::train(relationship, data = training.temp, method = "svmRadial", 
                                       trControl = fitControl, tuneGrid = tuneGrid,
                                       verbose = F, preProc = c("center", "scale"), metric = "ROC"), 
                  "rda" = caret::train(relationship, data = training.temp, method = "rda", 
                                       trControl = fitControl, tuneGrid = tuneGrid,
                                       verbose = F, metric = "ROC"),
                  "pls" = caret::train(relationship, data = training.temp, method = "pls", 
                                       trControl = fitControl, tuneGrid = tuneGrid,
                                       verbose = F, preProc = c("center", "scale"), metric = "ROC"))
    
    newlambda <- findlambda(Fit, newdata=testing.temp, Y=testing.tempY, rho=rho, save.pdf=FALSE, show.plot=FALSE)
    newlambda$method <- modelName
    res <- rbind(res, newlambda)
  }
  
  stopCluster(cl)
  toc()
  return(res)
}

# find rho
findrho <- function(Y, class0name=0) {
  class_names <- sort(names(table(Y)))
  if (class0name == 0) class0name <- class_names[1]
  class1name <- class_names[which(class_names != class0name)]

  count_table <- table(Y)
  rho <- count_table[class0name]/sum(count_table)

  attr(rho, "names") <- "rho"
  attr(rho, "Positive") <- class0name
  attr(rho, "Negative") <- class1name
  attr(rho, "Count") <- count_table[[class0name]]
  attr(rho, "Total") <- length(Y)

  return(rho)
}

# fit result with Fermi-Dirac distribution
findlambda <- function(Fit, newdata=df, Y=testingY, rho=rho, save.pdf=FALSE, show.plot=TRUE) {
  options(digits = 4)
  
  # prepare data frame
  if ('summaplus' %in% class(Fit)) methodName <- attr(Fit, "modelName")
  else methodName <- Fit$modelInfo$label

  N <- length(Y)
  probTrue <- as.numeric((Y == attr(rho, "Positive")))
  prob <- predict(Fit, newdata=newdata, rho=rho, type='prob')[[attr(rho, "Positive")]]
  res <- data.frame(itemN = 1:length(prob), prob = prob, probT = probTrue)
  res <- res[order(-res$prob), ]
  res$rank <- 1:length(prob)
  
  # fit with initial values
  l1 <- median(res$rank)
  m <- NULL
  try(m <- nls(probT ~ I(1/(1+exp(l1-l2*rank))), data = res, start = list(l1=l1, l2=1)))

  if (!is.null(m)) {
    l1 <- coef(m)[[1]]
    l2 <- -coef(m)[[2]]
  }
  # calculate ROC
  n1 <- sum(res$probT == 1)
  n0 <- sum(res$probT == 0)
  AUROC <- (sum(res[res$probT == 0, 'rank'])/n0 - sum(res[res$probT == 1, 'rank'])/n1)/length(prob) + 0.5
  temp <- optim(c(N*rho*0.1, 0.1), costFunc, rho=rho, auroc=AUROC, N=N)
  l1 <- -temp$par[1]
  l2 <- temp$par[2]
  rStar <- 1/l2 * log((1 - rho)/rho) - l1/l2
  
  # plot result
  if (show.plot) {
    #dev.new(width=6, height=4)
    plot(res$rank, res$probT, xlab='Rank', ylab='P(1|r)', ylim=c(0,1), main=methodName)
    lines(res$rank, res$prob, pch=4, lwd=1)
    lines(res$rank, 1/(1+exp(l1 + l2*res$rank)), col="blue", lty=4, lwd=1)
    text(0, 0.9, sprintf("l1 = %.2f\nl2 = %.2f", l1, l2), adj=c(0,1))
    abline(v = rStar, col='gray')
    text(rStar, 0.1, sprintf("r* = %.2f ", rStar), adj=c(1,0))
    text(length(prob), 0.9, sprintf("AUROC = %.4f\n rho = %.4f", AUROC, rho), adj=c(1,1))

    if (!is.null(m)) {
      lines(res$rank, predict(m), col="red", lty=3, lwd=2)
    }

    # save pdf
    if (save.pdf) {
      filename <- paste0('rank_prob_', attr(Fit, "dataName"), '_', methodName, '.pdf')
      print(paste0('... save to ', filename))
      dev.copy(pdf, file=filename, width=8, height=6)
      dev.off()
    }
  }
  
  #summary(m)
  if (is.null(m)) return(NULL)
  else return(data.frame(l1 = l1, l2 = l2, rs = rStar))
}

# set up summaplus
summaplus_train <- function(..., dataName = "data") {
  res <- list(...)
  attr(res, "modelName") <- "SUMMA +"
  attr(res, "dataName") <- dataName
  attr(res, "class") <- "summaplus"

  return(res)
}

# S3 method for summaplus
predict.summaplus <- function(model.list, newdata = df, Y=testingY, rho = rho, type='all', verbose=FALSE) {
  # prepare output
  res <- data.frame(index = 1:length(newdata[, 1]))
  res$odd <- 0
  i <- 1

  # compute odd for all models
  for (model in model.list) {
    if (!('lambda' %in% names(model))) {
      model$lambda <- findlambda(model, newdata = newdata, Y=testingY, rho=rho, show.plot=FALSE)
    }
    rank <- rank(-predict(model, newdata = newdata, type='prob')[[attr(rho, "Positive")]])
    
    # show model info
    if (verbose) cat(sprintf("[%i] Model %s : rStar = %.4f l1 = %.4f l2 = %.4f\n", i, model$modelInfo$label, model$lambda['rs'], model$lambda['l1'], model$lambda['l2']))
    
    odd <- model$lambda[[2]]*(model$lambda[[3]] - rank)
    res$odd <- res$odd + odd
    i <- i + 1
  }

  # calculate probability
  res$prob <- ROCit::invlogit(res$odd)
  res$rank <- rank(res$odd)
  res$response <- attr(rho, "Negative")
  res[res$odd < 0, 'response'] <- attr(rho, "Positive")
  res$response <- as.factor(res$response)

  if (type == 'response') return (res$response)
  else if (type == 'prob') {
    class0name <- attr(rho, "Positive")
    class1name <- attr(rho, "Negative")
    res <- data.frame(x = res$prob, y = 1 - res$prob)
    names(res) <- c(class0name, class1name)
    return (res)
  }
  else return (res)
}

# compute AUC values for all models with same datat set
computeAUCs <- function(model.list, newdataX, newdataY, rho=rho, verbose=FALSE) {
  # chack arguments
  if (class(model.list) != 'list') model.list <- list(model.list)

  # check SUMMA+ 
  for (i in model.list)  if ("summaplus" %in% class(i))  addmodel <- i 
  model.list <- append(model.list, addmodel)

  res <- c()
  modelnames <- c()
  i <- 1

  for (model in model.list) {
    if ("summaplus" %in% class(model)) modelName <- attr(model, "modelName")
    else modelName <- model$modelInfo$label

    temp <- predict(model, newdata=newdataX, rho=rho, type='prob')[[attr(rho, "Positive")]]
    fit.roc <- rocit(score = temp, class = as.character(newdataY), negref = attr(rho, "Negative"))

    modelnames<- c(modelnames, modelName)
    res <- c(res, fit.roc$AUC)

    if (verbose) cat(paste0(modelnames[i], ' - AUC - ', res[i], '\n'))
    i <- i + 1
  }
  res <- data.frame(t(res))
  colnames(res) <- modelnames
  return(res)
}

# calculate lambda1 and lambda2 using FD distribution
costFunc <- function(l, rho, auroc, N) {
  r <- 1:N
  sum1 <- sum(1/(1+exp(l[2]*r - l[1])))/N
  sum2 <- sum(r/(1+exp(l[2]*r - l[1])))/(N*N*rho)

  (rho - sum1)^2 + (1 + .5/N - rho/2 - auroc*(1-rho) - sum2)^2
}

# calculate auroc from lambda1, lambda2 
computeAUC_fromlambda <- function(l, rho, N) {
  r <- 1:N
  sum1 <- sum(r/(1+exp(l[2]*r - l[1])))/(N*rho)
  sum2 <- sum(r*(1 - 1/(1+exp(x[2]*r - x[1]))))/(N*(1-rho))
  (sum2 - sum1)/N + .5
}
