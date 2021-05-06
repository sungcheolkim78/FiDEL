library(caret)
library(sjmisc)
library(data.table)
library(tidyverse)
library(FiDEL)
library(devtools)

library(doParallel)
library(tictoc)
library(pROC)

library(ggpubr)

#########################
#NAVIGATE TO KAGGLE DIRECTORY WITHIN FIDEL BEFORE EXECUTING THE FOLLOWING CODE
#######################
#setwd("~/Documents/FiDEL/kaggle")

load_all("../R")

set.seed(200)

#West Nile
train <- as.data.table(readr::read_csv('data/data-westnile.csv.bz2'))
train$y <- as.factor(train$y)
train <- train[, -c('X41', 'X48', 'X82', 'X84', 'X12', 'X24', 'X36', 'X60', 'X72', 'X83')]

folds <- createFolds(train$y, k=22, list = TRUE)
traininglist <- lapply(folds, function(x) train[x, ])
testing  <- traininglist[[22]]
testingY <- to_label(testing$y, class1='Yes')

model_list <- c('rmda', 'rotationForest', 'pls', 'rda', 'svmLinear', 'svmRadial', 'knn', 'earth', 
          'mlp', 'rf', 'gbm', 'ctree', 'C5.0', 'bayesglm', 'glm', 'glmnet', 'simpls', 'dwdRadial', 'xgbTree', 
          'xgbLinear', 'nnet')

t1 <- mtrainer(model_list, dataInfo = 'westnile')


t1 <- train.mtrainer(t1, y~., traininglist, update=T)

saveRDS(t1, "../../t1_westnile.rds")

t2 <- predict.mtrainer.train(t1, newdata2=traininglist, class1=NULL)

saveRDS(t2, "../../t2_westnile.rds")

auclist_train <- lapply(1:21, function(x) auc_rank_train(t2$predictions[,x], traininglist, t2$nmethods, x))

names(auclist_train) <- t2$model_list

auclist_train <- unlist(auclist_train)

fde3 <- fde(t2$predictions)

entire.train <- traininglist[-22]
nrow.train <- unlist(lapply(1:t1$nmethods, function(x) do.call(rbind, entire.train[-x]) %>% nrow()))

prevalence.train <- unlist(lapply(1:t1$nmethods, function(x) do.call(rbind, entire.train[-x]) %>% as_tibble() %>% mutate(p=ifelse(as.character(y)=="Yes", 1, 0)) %>% pull(p) %>% mean()))

fde4 <- predict_performance(fde3, auclist_train, prevalence.train, nrow.train)

t1.test <- predict.mtrainer(t1, newdata=testing)

fde4@predictions <- t1.test$predictions

testset.items <- fde(t1.test$predictions)

fde4@rank_matrix <- testset.items@rank_matrix

fde4@nsamples <- testset.items@nsamples

fde5 <- calculate_performance(fde4, testingY, "FiDEL")

#correlation between ranks of base classifiers of test set items
wnv.cor <- corrank(fde5)

#empirical probability of class given rank vector for each base classifier 
wnv.pcr <- fidel.fits(t1, traininglist)
#analytical probability of class given rank vector for each base classifier 
fd.wnv <- fd.coords(fde5, wnv.pcr)

fde.woc <- calculate_performance(fde4, testingY, "WoC")

#include WoC in calculations, compute overall performance
wnv.overall <- overall_performance(fde5, fde.woc, 3:10, 200, 100, 'SE')

g1wnv <- plot_performance(fde5, nmethod_list=c(3, 5, 7), nsample=200, filename='../../wnv_performance.pdf')

#plot of final performance
gwnv <- ggarrange(g1wnv, wnv.overall, labels=c('A', 'B'), ncol=2, nrow=1, widths = c(2.7,1))




