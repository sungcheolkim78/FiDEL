# pcr.R - probability of class y at given rank r
#
# Sungcheol Kim @ IBM
# 2020/03/12
#
# version 1.0

library(ggpubr)
library(pROC)
library(doParallel)

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

new_pcr <- function(scores, y, N=0, M=1) {
  if (N == 0) {
    N <- length(scores)
    M <- 1
  }

  # calculate metrics on rank
  rho <- attr(y, 'rho')
  N.data <- length(y)
  ranked_y <- y[order(scores)]
  tpr0 <- cumsum(as.numeric(ranked_y == attr(y, 'class1')))/(rho*N.data)
  fpr0 <- cumsum(as.numeric(ranked_y == attr(y, 'class2')))/((1-rho)*N.data)
  bac0 <- 0.5*(tpr0 + 1 - fpr0)
  r_bac <- which.max(bac0)/N.data
  auc0 <- auc.rank(scores, y)

  # calculation on pcr
  x <- cal_pcr(scores, y, N=N, M=M)
  N1 <- floor(N*rho)
  N2 <- N - N1
  pcrd <- x[[1]]
  pcrd$tpr <- cumsum(pcrd$prob)/N1
  pcrd$fpr <- cumsum(1-pcrd$prob)/N2
  pcrd$prec <- cumsum(pcrd$prob)/(1:N)
  pcrd$bac <- 0.5*(pcrd$tpr + 1 - pcrd$fpr)

  auc_pcr <- 2*mean(pcrd$bac) - 0.5
  auprc_pcr <- (0.5*rho*(1 + sum(pcrd$prec[1:(N-2)] * pcrd$prec[2:(N-1)])/(N*rho*rho)))
  b <- get_fermi(auc_pcr, rho=attr(y, 'rho'), N=N)
  b0 <- get_fermi(auc0, rho=attr(y, 'rho'))

  # calculate metrics on score and label data
  roc_all <- auc.rank(scores, y)
  #ci0 <- ci(roc_all)
  #sigma0 <- (ci0[[2]]-ci0[[1]])/(2*1.96)

  structure(pcrd,
            N=N,
            M=M,
            N.data=length(scores),
            N1=N1,
            N2=N2,
            rho=attr(y, 'rho'),
            class1=attr(y, 'class1'),
            class2=attr(y, 'class2'),
            auclist=x[[2]],
            auc_sample=mean(x[[2]]),
            sigma_sample=sd(x[[2]]),
            auc0=auc0,
            ci0=0,
            sigma0=0,
            auc_pcr=auc_pcr,
            auprc_pcr=auprc_pcr,
            beta=b[[1]],
            mu=b[[2]],
            rs=b[[3]],
            r_th=r_bac,
            rs0=b0[[3]],
            bac0=bac0,
            class=c("pcr", "data.table", "data.frame"))
}

validate_pcr <- function(x) {
  # check binary classes

  x
}

pcr <- function(scores, y, N=0, M=1) {
  validate_pcr(new_pcr(scores, y, N=N, M=M))
}

cal_pcr <- function(scores, y, N=0, M=100) {
  stopifnot(N <= length(scores))

  # without sampling
  if (N == 0) {
    prob <- as.double(y == attr(y, 'class1'))
    rank <- frankv(scores, order=-1L)        # add rank
    auclist <- c(auc.rank(scores, y))

    return (list(data.table(rank=rank, prob=prob), auclist))
  } else {
    # with sampling
    prob <- vector(length = N)
    auclist <- vector(length = M)
    N1 <- floor(attr(y, 'rho')*N)
    N2 <- N - N1
    totalidx <- 1:length(scores)
    c1_idx <- totalidx[y == attr(y, 'class1')]
    c2_idx <- totalidx[y != attr(y, 'class1')]

    for (i in 1:M) {
      # create index from class1 and class2
      idx <- c(sample(c1_idx, N1), sample(c2_idx, N2))
      auclist[i] <- auc.rank(scores[idx], y[idx])

      temp <- as.double(y[idx] == attr(y, 'class1'))
      prob <- prob + temp[order(scores[idx], decreasing = FALSE)]
    }
    prob <- prob/M
    #sds <- sqrt(rowSums((res - rowMeans(res))^2)/(dim(res)[2] - 1))

    return (list(data.table(rank=seq(1,N,by=1.), prob=prob), auclist))
  }
}

get_pcr <- function(pcrd) {
  return (list(N0 = attr(pcrd, 'N.data'),
                N = attr(pcrd, 'N'),
                M = attr(pcrd, 'M'),
                rho = attr(pcrd, 'rho'),
                auc0 = attr(pcrd, 'auc0'),
                auc_sample = attr(pcrd, 'auc_sample'),
                auc_pcr = attr(pcrd, 'auc_pcr'),
                beta = attr(pcrd, 'beta'),
                mu = attr(pcrd, 'mu'),
                sigma0 = attr(pcrd, 'sigma0'),
                rs = attr(pcrd, 'rs'),
                rs0 = attr(pcrd, 'rs0'),
                r_th = attr(pcrd, 'r_th')))
}

print.pcr <- function(pcrd) {
  return(get_pcr(pcrd))
}

plot.pcr <- function(pcr_data) {
  df <- data.table(x=pcr_data$rank, y=pcr_data$prob, sd=pcr_data$sd/sqrt(attr(pcr_data, 'M')))
  auclist <- attr(pcr_data, 'auclist')

  l <- nls(y ~ fermi.b(x, b, m), data=df, start = list(b=attr(pcr_data, 'beta'), m=attr(pcr_data, 'mu')))
  l1 <- coef(l)
  fy <- fermi.b(df$x, attr(pcr_data, 'beta'), attr(pcr_data, 'mu'))

  msg1 <- sprintf("Mean AUC: %.4f\nStd. of AUC: %.4f", mean(auclist), sd(auclist))
  msg2 <- sprintf("beta: %.4f\nmu: %.4f", attr(pcr_data, 'beta'), attr(pcr_data, 'mu'))
  msg3 <- sprintf("N0: %d\nN: %d\nM: %d", attr(pcr_data, 'N.data'), attr(pcr_data, 'N'), attr(pcr_data, 'M'))

  g <- ggplot(data=df) + geom_point(aes(x=x, y=y)) +
    geom_line(aes(x=x, y=predict(l)), linetype="dashed", color="red") +
    geom_line(aes(x=x, y=fy), linetype="dashed", color="green") +
    geom_errorbar(aes(x=x, ymin=y-sd, ymax=y+sd), alpha=0.5, position=position_dodge(0.05)) +
    theme_classic() + ylab('P(1|r)') + xlab('Rank') +
    annotate("text", label=msg1, x=0, y=0, hjust=0, vjust=0) +
    annotate("text", label=msg2, x=mean(pcr_data$rank), y=0.4, vjust=1, hjust=1) +
    annotate("text", label=msg3, x=max(pcr_data$rank), y=1.0, hjust=1, vjust=1)

  ggsave(paste0("N", attr(pcr_data, 'N'), "M", attr(pcr_data, 'M'), ".pdf"), width=7, height=4)
  return (g)
}

auc.pcr <- function(pcr_data) {
  # rank and prob
  res <- abs(sum(pcr_data$rank * pcr_data$prob)/attr(pcr_data, "N1") -
               sum(pcr_data$rank * (1 - pcr_data$prob))/attr(pcr_data, "N2"))/attr(pcr_data, "N") + 0.5
  return (res)
}

auprc.pcr <- function(scores) {
  check <- c("prob", "prec") %in% names(scores)
  if(!all(check)) {
    scores <- score.to.classprob(scores)
    scores <- cal.fromRank(scores)
  }

  N <- length(scores$prob)
  rho <- sum(scores$prob)/N
  res <- sum(scores$prec[1:(N-2)] * scores$prec[2:(N-1)])/N
  return (0.5*rho*(1 + res/(rho*rho)))
}

sigma.rank <- function(rankprob, debug.flag = FALSE) {
  check <- c("rank", "prob") %in% names(rankprob)
  stopifnot(all(check))

  N1 <- sum(rankprob$prob)
  N2 <- length(rankprob$prob) - N1
  rmean <- sum(rankprob$rank * rankprob$prob)/N1
  r2mean <- sum(rankprob$rank * rankprob$rank * rankprob$prob)/N1

  if (debug.flag) {
    print(paste0('N: ', length(rankprob$prob)))
    print(paste0('N1: ', sum(rankprob$prob)))
    print(paste0('variance <r|1>: ', r2mean - rmean*rmean))
    print(paste0('sigma <r|1>: ', sqrt(r2mean - rmean*rmean)))
    print(paste0('sigma AUC: ', sqrt(r2mean - rmean*rmean)/N2))
  }

  return (sqrt(r2mean - rmean*rmean))
}
