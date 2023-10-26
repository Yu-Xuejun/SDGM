# Carry out MCMC on simulated data
rm(list = ls())
library(rstan)
library(moments)
library(Rlab)
library(LaplacesDemon)
set.seed(1)

prediction_evaluation <- function(y_test, X, beta, z){
  n_samples = dim(beta)[1]
  pll = rep(0,n_samples)
  acc = rep(0,n_samples)
  
  for (k in 1:n_samples){
    logits = X %*% beta[k,] + z[k,]
    prob_y = invlogit(logits)
    
    plls = dbern(y_test, prob_y, log = TRUE)
    pll[k] = mean(plls)
    
    y_pred = (prob_y > 0.5) * 1
    acc[k] = sum(y_test == y_pred) / length(y_test)
  }
  return(list(pll=mean(pll),acc=mean(acc)))
}




# function to do MCMC on simulated data
Simulation_MCMC <- function(n,R){
  
  mu_mcmc = matrix(NA, nrow=R, ncol=n)
  sigma_mcmc = matrix(NA, nrow=R, ncol=n)
  k_mcmc = matrix(NA, nrow=R, ncol=n)
  pll = rep(NA,R)
  acc = rep(NA,R)
  

  for (r in 1:R) {
    pred_name=paste("wheeze-test-",n,"-",r,".rds",sep = "")
    pred_data = readRDS(file=pred_name)
    
    name = paste("wheeze-",n,"-",r,".rds",sep = "")
    data = readRDS(file=name)
    fit <- stan(file = 'model_wheeze.stan',
                data = data, iter = 50000, chains = 1, thin=5)
    la <- rstan::extract(fit, permuted = TRUE, inc_warmup=FALSE) # return a list of arrays
    mu_mcmc[r,] = colMeans(la$b)
    sigma_mcmc[r,] = apply(la$b,2,sd)
    k_mcmc[r,] = apply(la$b,2,skewness)
    
    rst = prediction_evaluation(pred_data$y_test, pred_data$X, la$beta, la$b)
    pll[r] = rst$pll
    acc[r] = rst$acc
  }

  
  mcmc_moments <- list(mu_mcmc=mu_mcmc,sigma_mcmc=sigma_mcmc,k_mcmc=k_mcmc,pll=pll,acc=acc)
  name_mcmc = paste("wheeze-",n,"-",r,"-mcmc.rds",sep = "")
  saveRDS(mcmc_moments,file=name_mcmc)
  print("Done")
  
}

start_time <- Sys.time()
# Simulation_MCMC(n=50,R=10) # r=3 MCMC convergence problem
# Simulation_MCMC(n=100,R=10)
# Simulation_MCMC(n=250,R=10)
Simulation_MCMC(n=750,R=10)
end_time <- Sys.time()
print(end_time - start_time)





