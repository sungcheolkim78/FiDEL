# Kaggle Competition

In this folder, the examples of using FiDEL package for the Kaggle competition are saved. In general, you can download the training data for each competition on the Kaggle homepage. And you can also find the initial or preliminary data preprocessing steps in the Jupiter notebook files for each data set. After some basic imputing and cleaning in this preprocessing step, we use the caret package to train the selected list of models. Then, we use the training set performance to estimate the FiDEL parameters. Finally, we apply our method, FiDEL, and compare it to the Wisdom of Crowds ensembling technique, and the best individual base classifier. The final plots for the two datasets can be found in `results/wnv_performance.pdf` and `results/slm_performance.pdf`.  

As an example, we will consider the West Nile Virus Kaggle dataset and diagram a workflow. The entire analysis flow can be found in `westnile_FiDEL.R`

# Data preprocessing

## West Nile Virus

- You can find the jupyter notebook for the preprocessing in `kaggle-2-WestNile-data-prep.ipynb`
- The final dataset is saved as `data/data-westnile.csv.bz2`

Note that all of the custom functions used here can be found in the `FiDEL/R` directory. 

Load data and remove zero variance columns
```{r}
train <- as.data.table(readr::read_csv('data/data-westnile.csv.bz2'))
train$y <- as.factor(train$y)
train <- train[, -c('X41', 'X48', 'X82', 'X84', 'X12', 'X24', 'X36', 'X60', 'X72', 'X83')]
```

Divide data set with 22 groups
```{r}
set.seed(200)
folds <- createFolds(train$y, k=22, list = TRUE)
traininglist <- lapply(folds, function(x) train[x, ])
testing  <- traininglist[[22]]
testingY <- to_label(testing$y, class1='Yes')
```

# Training the models

Select model names and create multi-trainer (mtrainer) object.
```
model_list <- c('rmda', 'rotationForest', 'pls', 'rda', 'svmLinear', 'svmRadial', 'knn', 'earth', 
          'mlp', 'rf', 'gbm', 'ctree', 'C5.0', 'bayesglm', 'glm', 'glmnet', 'simpls', 'dwdRadial', 'xgbTree', 
          'xgbLinear', 'nnet')
t1 <- mtrainer(model_list, dataInfo = 'westnile')
```

Train all algorithms with list of group data sets. Note: this may take a few minutes depending on your computing power. 
```{r}
t1 <- train.mtrainer(t1, y~., traininglist, update=T)
```

We need to now calculate the performance of each base classifier on the rest of the training samples not used to learn that particular classifier. 
```{r}
t2 <- predict.mtrainer.train(t1, newdata2=traininglist, class1=NULL)

auclist_train <- lapply(1:21, function(x) auc_rank_train(t2$predictions[,x], traininglist, t2$nmethods, x))

names(auclist_train) <- t2$model_list

auclist_train <- unlist(auclist_train)
```

# Estimating the FiDEL parameters

Now, we need to estimate the FiDEL parameters, namely beta and mu for each of the base classifiers. We will use the training set metrics to do this. 
```{r}
fde3 <- fde(t2$predictions)

entire.train <- traininglist[-22]
nrow.train <- unlist(lapply(1:t1$nmethods, function(x) do.call(rbind, entire.train[-x]) %>% nrow()))

prevalence.train <- unlist(lapply(1:t1$nmethods, function(x) do.call(rbind, entire.train[-x]) %>% as_tibble() %>% mutate(p=ifelse(as.character(y)=="Yes", 1, 0)) %>% pull(p) %>% mean()))

fde4 <- predict_performance(fde3, auclist_train, prevalence.train, nrow.train)
```

# Base Classifier Performance on Test Set

Let us now predict the class labels on the test set data.  
```{r}
t1.test <- predict.mtrainer(t1, newdata=testing)

fde4@predictions <- t1.test$predictions

testset.items <- fde(t1.test$predictions)

fde4@rank_matrix <- testset.items@rank_matrix

fde4@nsamples <- testset.items@nsamples
```

# FiDEL performance

Next, let us estimate the class labels using FiDEL and the overall ensemble FiDEL performance. 
```{r}
fde5 <- calculate_performance(fde4, testingY, "FiDEL")
```

# Performance Comparison and Visualization

Finally, we can visualize our performance (`wnv.overall`). And, we can compare it to two standard methods: 1) Wisdom of Crowds (WoC), 2) Best Individual Classifier
```{r}
fde.woc <- calculate_performance(fde4, testingY, "WoC")
wnv.overall <- overall_performance(fde5, fde.woc, 3:10, 200, 100, 'SE')
```

There are a number of downstream things we can do now:    

1) compute the correlation between ranks of base classifiers of test set items
```{r}
wnv.cor <- corrank(fde5)
```

2) Compute the empirical probability of the class given rank vector for each base classifier 
```{r}
wnv.pcr <- fidel.fits(t1, traininglist)
```

3) And, the analytical probability of class given rank vector for each base classifier
```{r}
fd.wnv <- fd.coords(fde5, wnv.pcr)
```
Note that by combining items 2) and 3), we can compare the analytical and empirical probability of class given rank vectors. 

