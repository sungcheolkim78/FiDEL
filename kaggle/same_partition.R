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

############ WNV same partition ###########

train <- as.data.table(readr::read_csv('data/data-westnile.csv.bz2'))
train$y <- as.factor(train$y)
train <- train[, -c('X41', 'X48', 'X82', 'X84', 'X12', 'X24', 'X36', 'X60', 'X72', 'X83')]

folds <- createFolds(train$y, k=22, list = TRUE)
traininglist <- lapply(folds, function(x) train[x, ])
testing  <- traininglist[[22]]
testingY <- to_label(testing$y, class1='Yes')

model_list <- c('rmda', 'rotationForest', 'pls', 'rda', 'svmLinear', 'svmRadial', 'knn', 'earth', 'mlp', 'rf', 'gbm', 'ctree', 'C5.0', 'bayesglm', 'glm', 'glmnet', 'simpls', 'dwdRadial', 'xgbTree', 'xgbLinear', 'nnet')

t1 <- mtrainer(model_list, dataInfo = 'westnile')

t1 <- train.mtrainer.same.partition(t1, y~., traininglist, update=T)

saveRDS(t1, "../../same_partition_t1.wnv.rds")


t2 <- predict.mtrainer.train.same.partition(t1, newdata2=traininglist, class1=NULL)

auclist_train <- lapply(1:21, function(x) auc_rank_train_same_partition(t2$predictions[,x], traininglist, t2$nmethods, x))

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

fde.woc <- calculate_performance(fde4, testingY, "WoC")

wnv.overall <- overall_performance(fde5, fde.woc, 3:10, 200, 100, 'SE')

wnv.cor2 <- corrank(fde5) %>% mutate(Dataset="SLM", method="Same Partition")

############ SLM same partition ###########

train <- as.data.table(readr::read_csv('data/data-springleaf.csv.bz2'))
train$y <- ifelse(train$target == 1, 'Yes', 'No')
train$y <- as.factor(train$y)
train <- train[, -c('ID', 'target1', 'VAR_1427', 'VAR_0847', 'VAR_1428', 'VAR_0924')]

train <- train %>% 
  select(where(not_any_na)) 

feat_names <- colnames(train[,-c('y')])
rm_names <- c()

count <- 0
for (f in feat_names) {
  coldata <- train[[f]]
  if (any(coldata < 0)) {
    #print(paste0(f, '-', min(coldata)))
    count <- count + 1
    rm_names <- c(rm_names, f)
    next
  }
  if (any(coldata > 999999990)) {
    #print(paste0(f, '-', max(coldata)))
    count <- count + 1
    rm_names <- c(rm_names, f)
    next
  }
  if (sum(coldata > 9990 & coldata < 9999) > 20) {
    #print(paste0(f, '-', max(coldata)))
    count <- count + 1
    rm_names <- c(rm_names, f)
    next
  }
  if (sum(coldata > 990 & coldata < 999) > 20) {
    #print(paste0(f, '-', max(coldata)))
    count <- count + 1
    rm_names <- c(rm_names, f)
    next
  }
  if (sum(coldata > 90 & coldata < 99) > 20) {
    #print(paste0(f, '-', max(coldata)))
    count <- count + 1
    rm_names <- c(rm_names, f)
    next
  }
}

train_new <- train[,-rm_names, with=F]

folds <- createFolds(train_new$y, k=22, list = TRUE)
traininglist <- lapply(folds, function(x) train_new[x, ])
testing  <- traininglist[[22]]
testingY <- to_label(testing$y, class1='Yes')

model_list <- c('rmda', 'pls', 'rda', 'svmLinear', 'svmRadial', 'knn', 'earth', 'mlp', 'rf', 'ctree', 'C5.0', 'bayesglm', 'glm', 'glmnet', 'simpls', 'dwdRadial', 'xgbTree', 'xgbLinear', 'nnet', 'gbm')

t1.slm <- mtrainer(model_list, dataInfo = 'SpringLeaf')

t1.slm <- train.mtrainer.slm.same.partition(t1.slm, y~., traininglist, update=F)

saveRDS(t1.slm, "../../same_partition_t1.slm.rds")

t2.slm <- predict.mtrainer.train.same.partition(t1.slm, newdata2=traininglist, class1=NULL)
auclist_train_slm <- lapply(1:t2.slm$nmethods, function(x) auc_rank_train_same_partition(t2.slm$predictions[,x], traininglist,  t2.slm$nmethods, x))
names(auclist_train_slm) <- t2.slm$model_list
auclist_train_slm <- unlist(auclist_train_slm)

fde3.slm <- fde(t2.slm$predictions)

entire.train <- traininglist[-c(1, 21, 22)]
nrow.train <- rep(dim(do.call(rbind, entire.train))[1], 20)

prevalence.train <- rep(do.call(rbind, entire.train) %>% as_tibble() %>% mutate(p=ifelse(as.character(y)=="Yes", 1, 0)) %>% pull(p) %>% mean(), 20)


fde4.slm <- predict_performance(fde3.slm, auclist_train_slm, prevalence.train, nrow.train)

t1.test.slm <- predict.mtrainer(t1.slm, newdata=testing)

fde4.slm@predictions <- t1.test.slm$predictions

testset.items <- fde(t1.test.slm$predictions)

fde4.slm@rank_matrix <- testset.items@rank_matrix

fde4.slm@nsamples <- testset.items@nsamples

fde5.slm <- calculate_performance(fde4.slm, testingY, "FiDEL")
fde.woc.slm <- calculate_performance(fde4.slm, testingY, "WoC")


slm.overall <- overall_performance(fde5.slm, fde.woc.slm, 3:10, 200, 100, 'SE')

slm.cor2 <- corrank(fde5.slm) %>% mutate(Dataset="SLM", method="Same Partition")