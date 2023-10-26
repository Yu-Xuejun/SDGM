### The code below is modified from code used in the paper 
### "Gaussian variational approximation with sparse precision
### matrices" by Linda S. L. Tan and David J. Nott.
### (https://doi.org/10.1007/s11222-017-9729-7) 
### and kindly supplied by the authors of the paper.

library(tidyr)
library(dplyr)
library(matlib)
library(rstan)
df <-  read.table("data_epilepsy.txt", header=TRUE)
df_longer = pivot_longer(df,cols=c("Y1","Y2","Y3","Y4"),names_to="visit", values_to="Y")

df_longer$visit[df_longer$visit=="Y1"] <- -0.3
df_longer$visit[df_longer$visit=="Y2"] <- -0.1
df_longer$visit[df_longer$visit=="Y3"] <- 0.1
df_longer$visit[df_longer$visit=="Y4"] <- 0.3

df_longer <- df_longer %>% mutate(trt = ifelse(as.character(Trt) == "placebo", 0, 1)) %>%
  mutate(lb4 = log(Base4), clage = scale(log(Age),scale=FALSE)) %>%
  mutate(lb4trt = lb4*trt)

df_longer$trt = as.factor(df_longer$trt)
df_longer$visit = as.numeric(df_longer$visit)


(gm1 <- glmer(Y ~ lb4 * trt + clage + visit + (1 + visit| ID),
              data = df_longer, family = poisson))

saveRDS(gm1@beta, file = "beta0.rds")
saveRDS(gm1@theta, file = "zeta0.rds")

beta0 <- readRDS("beta0.rds")
zeta0 <- readRDS("zeta0.rds")

D <- matrix(c(zeta0[1],zeta0[2],zeta0[2],zeta0[3]),2,2)

X_R <- cbind(rep(1,236), df_longer$visit)
W <- matrix(0,59,4)
W_i <- NULL
If <- NULL
for(i in 1:59){
  If_ij = matrix(c(0,0,0,0),2,2)
  for(j in 1:4){
    If_ij <- If_ij + df_longer$Y[(i-1)*4+j] * t(t(X_R[(i-1)*4+j,])) %*% X_R[(i-1)*4+j,]
  }
  If_i = If_ij
  W_i = inv(If_i + inv(D)) %*% inv(D)
  W[i,] = W_i
}

a = X_R %*% W_i
C = matrix(0,2,6)
#C[:2,:2] = diag(2)
a = X_R %*% W_i %*% C

lb4 <- log(df$Base4)
trt <- c(rep(0,28),rep(1,31))
lage <- log(df$Age)
clage <- scale(lage,scale=FALSE)
V4 <- c(0,0,0,1)
visit <- c(-3,-1,1,3)/10
lb4trt <- lb4*trt
N <- sum(vni)
startindex <- c(0, cumsum(vni)[1:(n-1)]) + 1
endindex <- cumsum(vni)

################
# random slope #
################
for (i in 1:n){
  y[[i]] <- c(df$Y1[i], df$Y2[i], df$Y3[i], df$Y4[i])
  Z[[i]] <- cbind(rep(1,vni[i]), visit)
  X[[i]] <- cbind(rep(1,vni[i]), rep(lb4[i],4), rep(trt[i],4), rep(lb4trt[i],4), rep(clage[i],4), visit)
}
n <- length(y)
k <- dim(X[[1]])[2]
p <- dim(Z[[1]])[2]
d <- n*p + k + p*(p+1)/2
yall <- unlist(y)
Xall <- matrix(unlist(lapply(X,t)), ncol=k, byrow=TRUE)
Zall <- matrix(unlist(lapply(Z,t)), ncol=p, byrow=TRUE)
pzeros <- rep(0,p)
peye <- diag(p)
zetalength <- p*(p+1)/2
data <- list(n=n, N=as.integer(N), k=k, p=p, y=yall, X=Xall, Z=Zall, pzeros=pzeros, peye=peye,
             W=W, X_R=X_R,
             startindex=as.integer(startindex), endindex=as.integer(endindex), zetalength=as.integer(zetalength))

rstan_options(javascript=FALSE)
# stan (mcmc) #
start_time <- Sys.time()
fit <- stan(file = 'model_epilepsy_slope_t.stan',
            data = data, iter = 50000, chains = 1,thin=1)
end_time <- Sys.time()
print(end_time - start_time)

#print(fit, digits = 3)

la <- extract(fit, permuted = TRUE, inc_warmup=FALSE) # return a list of arrays 

#write.csv(la, "EpilepsyMCMC_t.csv")

#saveRDS(data, file = "EpilepsySlopeData_t.rds")



