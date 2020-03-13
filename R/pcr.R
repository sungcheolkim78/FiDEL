# pcr.R - probability of class y at given rank r
#
# Sungcheol Kim @ IBM
# 2020/03/12
#
# version 1.0

library(ggpubr)

new_pcr <- function(scores, y, N=0, M=1) {
  if (N == 0) {
    N <- length(scores)
    M <- 1
  }

  # calculate metrics on rank and pcr
  rho <- attr(y, 'rho')
  x <- cal_pcr(scores, y, N=N, M=M)
  N1 <- floor(N*rho)
  N2 <- N - N1
  pcrd <- x[[1]]
  pcrd$tpr <- cumsum(pcrd$prob)/sum(pcrd$prob)
  pcrd$fpr <- cumsum(1-pcrd$prob)/sum(1-pcrd$prob)
  pcrd$prec <- cumsum(pcrd$prob)/(1:N)
  pcrd$bac <- 0.5*(pcrd$tpr + 1 - pcrd$fpr)

  auc_pcr <- abs(sum(pcrd$rank * pcrd$prob)/N1 - sum(pcrd$rank * (1 - pcrd$prob))/N2)/N + 0.5
  auprc_pcr <- (0.5*rho*(1 + sum(pcrd$prec[1:(N-2)] * pcrd$prec[2:(N-1)])/(N*rho*rho)))
  l <- lambda.auc(auc_pcr, N=N, rho=attr(y, 'rho'))

  # calculate metrics on score and label data
  roc_all <- roc(y, scores)
  ci0 <- ci(roc_all)
  sigma0 <- (ci0[[2]]-ci0[[1]])/(2*1.96)

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
            auc0=auc(roc_all),
            ci0=ci0,
            sigma0=sigma0,
            auc_pcr=auc_pcr,
            auprc_pcr=auprc_pcr,
            l1=l[[1]],
            l2=l[[2]],
            rs=l[[3]],
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
    res <- matrix(0, nrow=N, ncol=M)
    auclist <- numeric(0)
    N1 <- floor(attr(y, 'rho')*N)
    N2 <- N - N1
    totalidx <- 1:length(scores)
    c1_idx <- totalidx[y == attr(y, 'class1')]
    c2_idx <- totalidx[y != attr(y, 'class1')]

    for (i in 1:M) {
      # keep rho ratio or not

      idx1 <- sample(c1_idx, N1)
      idx2 <- sample(c2_idx, N2)
      temp <- data.table(score=scores[c(idx1, idx2)], y=y[c(idx1, idx2)])
      temp <- temp[order(temp$score, decreasing = FALSE), ]

      auclist <- c(auclist, auc.rank(temp$score, temp$y))

      res[, i] <- as.double(temp$y == attr(y, 'class1'))
    }
    prob <- rowMeans(res)
    sds <- sqrt(rowSums((res - rowMeans(res))^2)/(dim(res)[2] - 1))
    rank <- 1:N

    return (list(data.table(rank=rank, prob=prob, sd=sds), auclist))
  }
}

print.pcr <- function(pcrd) {
  res <- data.table(N0 = attr(pcrd, 'N.data'),
                    N = attr(pcrd, 'N'),
                    M = attr(pcrd, 'M'),
                    rho = attr(pcrd, 'rho'),
                    auc0 = attr(pcrd, 'auc0'),
                    auc_sample = attr(pcrd, 'auc_sample'),
                    auc_pcr = attr(pcrd, 'auc_pcr'),
                    auc_pxysum = auc.Pxysum(pcrd),
                    auc_pxysample = auc.Pxysample(pcrd),
                    l1 = attr(pcrd, 'l1'),
                    l2 = attr(pcrd, 'l2'),
                    sigma0 = attr(pcrd, 'sigma0'),
                    sigma = attr(pcrd, 'sigma'))
  return(res)
}

plot.pcr <- function(pcr_data) {
  df <- data.table(x=pcr_data$rank, y=pcr_data$prob, sd=pcr_data$sd/sqrt(attr(pcr_data, 'M')))
  auclist <- attr(pcr_data, 'auclist')

  l <- nls(y ~ fermi(x, l1, l2), data=df, start = list(l1=attr(pcr_data, 'l1'), l2=attr(pcr_data, 'l2')))
  l1 <- coef(l)
  fy <- fermi(df$x, attr(pcr_data, 'l1'), attr(pcr_data, 'l2'))

  msg1 <- sprintf("Mean AUC: %.4f\nStd. of AUC: %.4f", mean(auclist), sd(auclist))
  msg2 <- sprintf("l1: %.4f\nl2: %.4f", attr(pcr_data, 'l1'), attr(pcr_data, 'l2'))
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


plot.curves <- function(pcrd) {
  idx <- which.max(pcrd$bac)
  auc_bac <- 2*sum(pcrd$bac)/length(pcrd$bac) - 0.5
  auc_rank <- attr(pcrd, 'auc0')
  N <- attr(pcrd, 'N')
  N1 <- attr(pcrd, 'N1')
  N2 <- attr(pcrd, 'N2')

  g1 <- ggplot(data=pcrd, aes(x=fpr, y=tpr)) + geom_point(alpha=0.7) +
    xlab("FPR") + ylab("TPR") + theme_classic() +
    geom_abline(intercept = 0, slope = 1, linetype="dashed") +
    annotate("point", x=pcrd$fpr[idx], y=pcrd$tpr[idx], color = "red", shape=8, size=3) +
    annotate("text", label=sprintf("AUC (rank): %.4f", auc_rank), x=1, y=0, hjust=1)
  g2 <- ggplot(data=pcrd, aes(x=tpr, y=prec)) + geom_point(alpha=0.7) +
    xlab("TPR") + ylab("Prec") + ylim(c(0,1)) + theme_classic() +
    geom_hline(yintercept = sum(pcrd$prob)/length(pcrd$prob), linetype="dashed") +
    annotate("point", x=pcrd$tpr[idx], y=pcrd$prec[idx], color = "red", shape=8, size=3) +
    annotate("text", x=1, y=0, hjust=1, vjust=0,
             label=sprintf("N: %.0f\nN1: %.0f\nN2: %.0f", N, N1, N2)) +
    annotate("text", label=sprintf("AUPRC: %.4f", attr(pcrd, 'auprc_pcr')),
             x=pcrd$tpr[idx]-0.05, y=pcrd$prec[idx], hjust=1, vjust=1)
  g3 <- ggplot(data=pcrd, aes(x=rank, y=bac)) + geom_point(alpha=0.7) +
    xlab("Rank") + ylab("Balanced Accuracy") + ylim(c(0.5, 1)) + theme_classic() +
    geom_vline(xintercept = pcrd$rank[idx], linetype="dashed") +
    annotate("text", label=sprintf("AUC (bac): %.4f", auc_bac), x=0, y=1, hjust=0) +
    annotate("text", label=sprintf("thr.: %i", idx), x=pcrd$rank[idx], y=0.5)

  g <- ggarrange(g1, g2, g3, labels=c("A", "B", "C"), ncol=3  , nrow=1)
  ggsave("temp.pdf", width=12, height=4, dpi=300)
  return (g)
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
