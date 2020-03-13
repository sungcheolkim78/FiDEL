# ranked.R - ranked probability based algorithms
#
# Sungcheol Kim @ IBM
# 2020/01/15
#
# version 1.0

library(data.table)
library(ggpubr)

# Using class proability at given rank
auc.rank <- function(scores, y, class1=NULL) {
  # score and label

  stopifnot(length(scores) == length(y))
  labels <- levels(y)
  if (is.null(class1)) {
    class1 <- labels[[1]]
  }

  # calculate class 1 and class 2
  N <- length(scores)
  N1 <- sum(y == class1)
  N2 <- N - N1
  mat <- data.table(scores=scores, y=y)
  mat$rank <- frankv(scores, order=-1)

  res <- abs(sum(mat$rank[y == class1])/N1 - sum(mat$rank[y != class1])/N2)/N + 0.5
  return(res)
}

# characteristics based on the rank threshold
cal.fromRank <- function(scores) {
  check <- c("rank", "prob") %in% names(scores)
  if(!all(check)) {
    scores <- score.to.classprob(scores)
  }

  N <- length(scores$rank)
  scores <- scores[order(scores$rank), ]
  scores$tpr <- cumsum(scores$prob)/sum(scores$prob)
  scores$fpr <- cumsum(1-scores$prob)/sum(1-scores$prob)
  scores$prec <- cumsum(scores$prob)/(1:N)
  scores$bac <- 0.5*(scores$tpr + 1 - scores$fpr)

  return(scores)
}

# calculate lambda1,2 from auc, rho
lambda.auc <- function(auc, N=100.0, rho=0.5, full=FALSE) {
  costFunc1 <- function(l, rho, rs, N) {
    r <- seq(1, N, by = 1.0)
    sum1 <- sum(1/(1+exp(l[2]*r - l[1])))
    sum2 <- sum(r/(N+N*exp(l[2]*r - l[1])))

    (N*rho - sum1)^2 + (rs/N - sum2)^2
  }

  costFunc2 <- function(l, rho, rs, N) {
    r <- seq(1, N, by = 1.0)
    sum1 <- sum(1/(1+exp(-l[2]*r + l[1])))
    sum2 <- sum(r/(N+N*exp(-l[2]*r + l[1])))
    rs_m <- N*(1-rho)*(N*(1-rho)+1)/2

    (N*(1-rho) - sum1)^2 + (0.5*(N+1) - rs/N - sum2)^2
  }

  # check dataframe
  check <- c('auc', 'N', 'rho') %in% names(auc)
  if (all(check)) {
    auroc <- auc$auc
    N <- auc$N
    rho <- auc$rho
  } else {
    auroc <- auc
  }
  N <- as.numeric(N)
  rs <- 0.5*N*rho*(N+1) - N*N*rho*(1-rho)*(auroc - 0.5)

  l <- lambda.appr(auroc, N=N, rho=rho)
  #print(l)
  initial <- c(l[['l1']], l[['l2']])

  temp <- optim(initial, costFunc1, rho=rho, rs=rs, N=N, control=list(maxit=8000, reltol=1e-12))
  if (temp$value > 1e-4) {
    temp <- optim(initial, costFunc2, rho=rho, rs=rs, N=N, control=list(maxit=8000, reltol=1e-12))
  }

  #print(temp)
  if (temp$value > 1e-5) message(sprintf("not converged - auc: %.4f, rho: %.4f", auroc, rho))
  l1 <- -temp$par[1]
  l2 <- temp$par[2]
  rs <- 1/l2 * log((1 - rho)/rho) - l1/l2

  if (full)
    return(data.table(l1 = l1, l2 = l2, rs = rs, l10 = initial[[1]], l20 = initial[[2]], auc=auroc, rho=rho))
  else
    return(c(l1 = l1, l2 = l2, rs = rs))
}

# calculate lambda1, lambda2 from auc, rho (version 2)
lambda.appr <- function(auc, N=100.0, rho=0.5) {
  N <- as.numeric(N)
  l1_low <- log(1/rho - 1) - 12*N*(auc-0.5)/(N*N-1)*((N+1+N*rho)*0.5 - N*rho*auc)
  l2_low <- 12*N*(auc-0.5)/(N*N-1)

  temp <- sqrt(rho*(1-rho)*(1-2*(auc-0.5)))
  l1_high <- -2*rho/(sqrt(3)*temp)
  l2_high <- 2/(sqrt(3)*N*temp)

  alpha <- 2*(auc - 0.5)
  l1 <- l1_high*alpha + l1_low*(1-alpha)
  l2 <- l2_high*alpha + l2_low*(1-alpha)
  rs <- 1/l2 * log((1 - rho)/rho) - l1/l2

  return(c(l1low=l1_low, l2low=l2_low, l1high=l1_high, l2high=l2_high, l1=l1, l2=l2, rs=rs))
}

# calculate beta, mu using normalized r
get_fermi <- function(auc, rho, resol=0.0001) {
  # cost function - normalized version
  # mu' = mu/N
  # beta' = beta*N
  # r*' = r*/N
  cost <- function(bm, auc, rho, resol) {
    rprime <- seq(0, 1, by=resol)
    sum1 <- sum(resol/(1+exp(bm[1]*(rprime-bm[2]))))
    sum2 <- sum(resol*rprime/(1+exp(bm[1]*(rprime-bm[2]))))

    (rho - sum1)^2 + (0.5*rho - rho*(1-rho)*(auc-0.5) - sum2)^2
  }

  # auc should be in (0.5, 1.0)
  stopifnot(auc > 0.5)
  stopifnot(auc <= 1.0)

  # calculate intital beta' and mu'
  l <- lambda.appr(auc, N=1000, rho=rho)

  # optimize cost function
  initial <- c(l[['l2']]*1000, -l[['l1']]/(1000*l[['l2']]))
  temp <- optim(initial, cost, auc=auc, rho=rho, resol=resol, control=list(maxit=8000, reltol=1e-12))

  rs <- 1/temp$par[1] * log((1 - rho)/rho) + temp$par[2]
  return(c(betap=temp$par[1], mup=temp$par[2], rsp=rs))
}

# plot histogram of score distribution
plot.scores <- function(scores, y) {
  y.flag <- TRUE
  if (missing(y)) {
    y <- rep('class1', length(scores))
    attr(y, 'class1') <- 'class1'
    attr(y, 'class2') <- 'class2'
    y.flag <- FALSE
  } else {
    stopifnot(all(c('factor', 'label') %in% class(y)))
  }
  df <- data.table(score=scores, y=y)

  score1 <- scores[y == attr(y, 'class1')]
  score2 <- scores[y == attr(y, 'class2')]

  g <- ggplot(data=df, aes(x=score, color=y)) + geom_histogram(alpha=0.5, position="identity", bins=50) +
    theme_classic() +
    theme(legend.position="top") +
    annotate("text", label=sprintf("mu1: %.4f", mean(score1)), x=mean(score1), y=0, vjust=0) +
    annotate("text", label=sprintf("N1: %d", length(score1)), x=mean(score1)-4*sd(score1), y=100, vjust=0)
  if (y.flag) {
    g <- g +
      annotate("text", label=sprintf("mu2: %.4f", mean(score2)), x=mean(score2), y=0, vjust=0) +
      annotate("text", label=sprintf("N2: %d", length(score2)), x=mean(score2)+4*sd(score2), y=100, vjust=0)
  }

  ggsave('score.pdf', width=8, height=6, dpi=300)
  return (g)
}

ensemble.fermi <- function(predictions, y, alpha = 1.0, method='+', debug.flag = FALSE) {
  M <- ncol(predictions)
  N <- nrow(predictions)

  res <- matrix(0, nrow=N, ncol=M)
  fd <- matrix(0, nrow=M, ncol=4)

  for(m in 1:M) {
    fd[m, 1] <- auc.rank(predictions[,m], y)
    #l <- lambda.auc(fd[m, 1], N=N, rho = attr(y, 'rho'))
    b <- get_fermi(fd[m, 1], attr(y, 'rho'))
    #fd[m, 2] <- l[['l1']]
    #fd[m, 3] <- l[['l2']]
    #fd[m, 4] <- l[['rs']]
    fd[m, 2] <- -b[[1]]*b[[2]]
    fd[m, 3] <- b[[1]]/N
    fd[m, 4] <- b[[3]]*N

    res[ , m] <- rank(predictions[,m])
    if (method == '+')
      res[ , m] <- l[['l2']]^alpha *(l[['rs']] - res[, m])
    else
      res[ , m] <- 12*N*(fd[m, 1] - 0.5)/(N*N - 1) *((N+1.)/2.- res[, m])
  }

  if(debug.flag) {
    #temp <- as.data.frame(res)
    #temp[['summa+']] <- rowMeans(res)
    #plot(temp)
    #print(cor(temp, method = "spearman"))
    print(fd)
  }

  return(rowSums(res))
}
