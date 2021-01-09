# Data preprocessing

## West Nile Virus

## Springleaf Marketing Response

# Training the models

```
model_list <- c('rmda', 'rotationForest', 'pls', 'rda', 'svmLinear', 'svmRadial', 'knn', 'earth', 'mlp', 'rf', 'gbm', 'ctree', 'C5.0', 'bayesglm', 'glm', 'glmnet', 'simpls', 'dwdRadial', 'xgbTree', 'xgbLinear', 'nnet')
t1 <- mtrainer(model_list, dataInfo = 'SpringLeaf')
```

```{r}
t1 <- train.mtrainer(t1, y~., traininglist, update=T)
```

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

```{r}
plot_performance(fde2, nmethod_list=c(3, 5, 7), nsample=200, filename='results/SLM_perf_fde2.pdf')
```

```{r}
plot_performance_nmethods(fde2, nmethod_list=3:10, nsample=200, filename='results/SLM_perf_nmethod_fde2.pdf')
```
