### The code below is modified from code used in the paper 
### "Conditionally structured variational Gaussian approximation 
### with importance weights" by Linda S. L. Tan, Aishwarya Bhaskaran 
### and David J. Nott. (https://doi.org/10.1007/s11222-020-09944-8) 
### and kindly supplied by the authors of the paper.

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


library(astsa)
data(nyse)
nyse_data = force(nyse)
log_rate_ratio = as.numeric(nyse_data)
n <- length(log_rate_ratio)
y = 100*(log_rate_ratio - mean(log_rate_ratio))
data <- list(n=n, y=y)
saveRDS(data, "data_nyse.rds")

# stan (mcmc) #
start_time <- Sys.time()
fit <- stan(file = 'model_svm.stan', seed=1077,
            data = data, iter = 50000, chains = 1, thin=1)
end_time <- Sys.time()
print(end_time - start_time)
# 36.0375 mins
print(fit)
la <- extract(fit) # return a list of arrays 

traceplot(fit,pars = "kappa",inc_warmup = TRUE)


write.csv(la, "NYSE_MCMC_FULL.csv")