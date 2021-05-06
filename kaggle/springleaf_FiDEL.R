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


#Spring Leaf Marketing
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

model_list <- c('rmda', 'pls', 'rda', 'svmLinear', 'svmRadial', 'knn', 'earth', 'mlp', 'rf', 'gbm', 'ctree', 'C5.0', 'bayesglm', 'glm', 'glmnet', 'simpls', 'dwdRadial', 'xgbTree', 'xgbLinear', 'nnet')


t1.slm <- mtrainer(model_list, dataInfo = 'SpringLeaf')

t1.slm <- train.mtrainer.slm(t1.slm, y~., traininglist, update=F)

saveRDS(t1.slm, "../../t1_springleaf.rds")

t2.slm <- predict.mtrainer.train(t1.slm, newdata2=traininglist, class1=NULL)

saveRDS(t2.slm, "../../t2_springleaf.rds")


auclist_train_slm <- lapply(1:t2.slm$nmethods, function(x) auc_rank_train(t2.slm$predictions[,x], traininglist,  t2.slm$nmethods, x))

names(auclist_train_slm) <- t2.slm$model_list

auclist_train_slm <- unlist(auclist_train_slm)

fde3.slm <- fde(t2.slm$predictions)

entire.train <- traininglist[-c(21, 22)]
nrow.train <- unlist(lapply(1:t1.slm$nmethods, function(x) do.call(rbind, entire.train[-x]) %>% nrow()))

prevalence.train <- unlist(lapply(1:t1.slm$nmethods, function(x) do.call(rbind, entire.train[-x]) %>% as_tibble() %>% mutate(p=ifelse(as.character(y)=="Yes", 1, 0)) %>% pull(p) %>% mean()))

fde4.slm <- predict_performance(fde3.slm, auclist_train_slm, prevalence.train, nrow.train)

t1.test.slm <- predict.mtrainer(t1.slm, newdata=testing)

fde4.slm@predictions <- t1.test.slm$predictions

testset.items <- fde(t1.test.slm$predictions)

fde4.slm@rank_matrix <- testset.items@rank_matrix

fde4.slm@nsamples <- testset.items@nsamples

fde5.slm <- calculate_performance(fde4.slm, testingY, "FiDEL")

#correlation matrix of the ranks of test set items of the base classifiers
slm.cor <- corrank(fde5.slm)

#empirical probability of class given rank vector for each base classifier 
slm.pcr <- fidel.fits(t1.slm, traininglist)
#analytical probability of class given rank vector for each base classifier 
fd.slm <- fd.coords(fde5.slm, slm.pcr)

fde.woc.slm <- calculate_performance(fde4.slm, testingY, "WoC")

#include WoC in calculations, and calculate overall performance
slm.overall <- overall_performance(fde5.slm, fde.woc.slm, 3:10, 200, 100, 'SE')

g1slm <- plot_performance(fde5.slm, nmethod_list=c(3, 5, 7), nsample=200, filename='../../slm_performance.pdf')

#plot of overall performance
gslm <- ggarrange(g1slm, slm.overall, labels=c('C', 'D'), ncol=2, nrow=1, widths = c(2.7,1))

 