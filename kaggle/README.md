# Kaggle Competition

In this folder, the examples of using FiDEL package for the Kaggle competition are saved. In general, you can download the training data for each competition in Kaggle homepage. And you can also find the initial or preliminary data preprocessing steps in `Notebooks` section. Here, we chose some of basic imputing and cleaning method for the dataset. After this process, we use the caret package to train the selected list of models. Then these models generate the AUC for the test data set and the FiDEL method creates the ensemble performace. 

# Data preprocessing

## West Nile Virus

- You can find the jupyter notebook for the preprocessing in `kaggle-2-WestNile-data-prep.ipynb`
- The final dataset is saved as `data/data-westnile.csv.bz2`

## Springleaf Marketing Response

- You can find the jupyter notebook for the preprocessing in `kaggle-4-Springleaf-data-prep.ipynb`
- The final dataset is saved as `data/data-springleaf.csv.bz2`

Load data file and set the target class and remove some unnecessary columns
```{r}
set.seed(200)
train <- as.data.table(readr::read_csv('data/data-springleaf.csv.bz2'))
train$y <- ifelse(train$target == 1, 'Yes', 'No')
train$y <- as.factor(train$y)
train <- train[, -c('ID', 'target1', 'VAR_1427', 'VAR_0847', 'VAR_1428', 'VAR0924')]
```

Clean up train data and remove columns with missing data (NA, 99999999X, 999X, 99X)
```{r}
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
```

Divide the total data in 22 groups to train each algorithm with different data group to minimize the correlation between the trained algorithms.
```{r}
set.seed(200)

folds <- createFolds(train_new$y, k=22, list = TRUE)
traininglist <- lapply(folds, function(x) train_new[x, ])
testing  <- traininglist[[22]]
testingY <- to_label(testing$y, class1='Yes')
```

# Training the models

Select model names and create multi-trainer (mtrainer) object.
```
model_list <- c('rmda', 'rotationForest', 'pls', 'rda', 'svmLinear', 'svmRadial', 'knn', 'earth', 
          'mlp', 'rf', 'gbm', 'ctree', 'C5.0', 'bayesglm', 'glm', 'glmnet', 'simpls', 'dwdRadial', 'xgbTree', 
          'xgbLinear', 'nnet')
t1 <- mtrainer(model_list, dataInfo = 'SpringLeaf')
```

Train all algorithms with list of group data sets.
```{r}
t1 <- train.mtrainer(t1, y~., traininglist, update=T)
```

Create the prediction with all included methods and AUC list.
```{r}
t1 <- predict.mtrainer(t1, newdata=testing)
auclist <- apply(t1$predictions, 2, auc_rank, testingY)
```

# Calculating the ensemble performance (FiDEL)

Without known labels, 
```{r}
fde1 <- fde(t1$predictions)
fde1 <- predict_performance(fde1, auclist, attr(testingY, 'rho'))
```

With known labels,
```{r}
fde2 <- calculate_performance(fde1, testingY)
```

Create plot with number of selected methods and iteration number.
```{r}
plot_performance(fde2, nmethod_list=c(3, 5, 7), nsample=200, filename='results/SLM_perf_fde2.pdf')
```
![](results/SLM_perf_fde2.pdf)

Create plot with different number of selected methods and iteration number.
```{r}
plot_performance_nmethods(fde2, nmethod_list=3:10, nsample=200, filename='results/SLM_perf_nmethod_fde2.pdf')
```
