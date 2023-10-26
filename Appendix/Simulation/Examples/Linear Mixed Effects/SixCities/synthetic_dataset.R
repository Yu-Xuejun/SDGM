# This file contains code to save different size of simulated synthetic datasets.
rm(list = ls())
library(geepack)
library(LaplacesDemon)
library(pracma)
library(dplyr)
library(tidyr)
library(stats)

set.seed(1)

data(ohio)

n = length(unique(ohio$id)) # no. of subjects
smoke_prob = sum(ohio$smoke)/(n*4)
ohio_new <- ohio

n_syn = 750-n
for (i in (n+1):(n+n_syn)) {
  smoke = rbinom(n=1,size=1,prob=smoke_prob)
  new_data <- data_frame(resp=rep(0,4),id=rep(i-1,4),age=c(-2,-1,0,1),smoke=rep(smoke,4))
  ohio_new <- rbind(ohio_new,new_data)
}
time <- rep(c(1,2,3,4),750)
ohio_wide <- ohio_new %>% mutate(time=time)
ohio_wide4 <- ohio_wide %>% pivot_wider(id_cols = id,names_from=time,values_from = c(age, smoke))
ohio_wide5 <- ohio_wide4 %>% mutate(age_5 = age_4+1, smoke_5=smoke_4)


ohio_long_age4 <- ohio_wide4 %>% select(!starts_with("smoke")) %>% pivot_longer(cols=starts_with("age"),names_to="time",names_prefix=c("age_"),values_to=c("age"))
ohio_long_smoke4 <- ohio_wide4 %>% select(!starts_with("age")) %>% pivot_longer(cols=starts_with("smoke"),names_to="time",names_prefix=c("smoke_"),values_to=c("smoke"))
ohio_long4 <- full_join(ohio_long_age4, ohio_long_smoke4,by = join_by(id, time))

ohio_long_age5 <- ohio_wide5 %>% select(!starts_with("smoke")) %>% pivot_longer(cols=starts_with("age"),names_to="time",names_prefix=c("age_"),values_to=c("age"))
ohio_long_smoke5 <- ohio_wide5 %>% select(!starts_with("age")) %>% pivot_longer(cols=starts_with("smoke"),names_to="time",names_prefix=c("smoke_"),values_to=c("smoke"))
ohio_long5 <- full_join(ohio_long_age5, ohio_long_smoke5,by = join_by(id, time))


# Simulate data
Simulate_data <- function(n, R, beta_posmean, zeta_posmean, ohio_long, timepoint=4){
  N = n*timepoint
  ID <- unique(ohio_long$id)
  idx <- sample(1:length(ID),size=n,replace = FALSE)
  ID_list = ID[idx]
  vni <- rep(timepoint,n)
  X <- NULL
  Z <- NULL
  age <- ohio_long$age
  smoke <- ohio_long$smoke
  smokeage <- ohio_long$smoke*ohio_long$age
  for (i in 1:n){
    rows <- which(ohio_long$id == ID_list[i])
    vni[i] <- length(rows)
    Z[[i]] <- cbind(rep(1,vni[i]))
    X[[i]] <- cbind(rep(1,vni[i]), smoke[rows], age[rows], smokeage[rows])
  }
  k <- dim(X[[1]])[2]                         # length of beta
  p <- dim(Z[[1]])[2]                         # length of b_i 
  d <- n*p + k + p*(p+1)/2
  Xall <- matrix(unlist(lapply(X,t)), ncol=k, byrow=TRUE)
  Zall <- unlist(Z)
  labels <- c('intercept','smoke','age','smoke x age','zeta')  
  N <- sum(vni)
  startindex <- c(0, cumsum(vni)[1:(n-1)]) + 1
  endindex <- cumsum(vni)
  N = as.integer(N)
  startindex = as.integer(startindex)
  endindex = as.integer(endindex)
  n = as.integer(n)
  for(r in 1:R){
    b_true = rnorm(n=n,mean = 0,sd=exp(-zeta_posmean))
    prob=rep(NA,N)
    for (i in 1:n) {
      for (j in startindex[i]:endindex[i]){
        prob[j] = dot(Xall[j,], beta_posmean) + Zall[j]*b_true[i]
      }
    }
    prob_y = invlogit(prob)
    Y_sim = rep(NA,N)
    for (i in 1:N) {
      Y_sim[i] = rbinom(n=1,size=1,prob_y[i])
    }
    
    # data <- list(n=n, N=N, k=k, y=Y_sim, X=Xall, Z=Zall, startindex=startindex, endindex=endindex)
    if(timepoint==5){
      y_test = Y_sim[endindex]
      X = Xall[endindex,]
      name=paste("wheeze-test-",n,"-",r,".rds",sep = "")
      data <- list(y_test=as.integer(y_test), X=X)
    }else if(timepoint==4){
      data <- list(n=n, N=N, k=k, y=Y_sim, X=Xall, Z=Zall, startindex=startindex, endindex=endindex)
      name=paste("wheeze-",n,"-",r,".rds",sep = "")
    }else{
      print("Error: invalid time point.")
    }
    saveRDS(data,file=name)
  }
  print("Done")
}

# posterior mean from MCMC (in real data analysis)
beta_posmean = c(-3.1022094,0.4622142,-0.2195170, 0.1073833)
zeta_posmean = -0.6777228


# simulation n=50, n=100, n=200, n=750
Simulate_data(n=50, R=10, beta_posmean, zeta_posmean, ohio_long4)
Simulate_data(n=100, R=10, beta_posmean, zeta_posmean, ohio_long4)
Simulate_data(n=250, R=10, beta_posmean, zeta_posmean, ohio_long4)
Simulate_data(n=750, R=10, beta_posmean, zeta_posmean, ohio_long4)

Simulate_data(n=50, R=10, beta_posmean, zeta_posmean, ohio_long5, timepoint=5)
Simulate_data(n=100, R=10, beta_posmean, zeta_posmean, ohio_long5, timepoint=5)
Simulate_data(n=250, R=10, beta_posmean, zeta_posmean, ohio_long5, timepoint=5)
Simulate_data(n=750, R=10, beta_posmean, zeta_posmean, ohio_long5, timepoint=5)