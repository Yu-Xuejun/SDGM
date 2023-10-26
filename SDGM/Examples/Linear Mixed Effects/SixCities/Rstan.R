### The code below is modified from code used in the paper 
### "Conditionally structured variational Gaussian approximation 
### with importance weights" by Linda S. L. Tan, Aishwarya Bhaskaran 
### and David J. Nott. (https://doi.org/10.1007/s11222-020-09944-8) 
### and kindly supplied by the authors of the paper.
library(geepack)
data(ohio)
age <- ohio$age
smoke <- ohio$smoke
smokeage <- ohio$smoke*ohio$age

ID <- unique(ohio$id)
n <- length(ID)                      # no. of subjects
vni <- rep(4,n)
X <- NULL
Z <- NULL
y <- NULL
for (i in 1:n){
    rows <- which(ohio$id == ID[i])
    vni[i] <- length(rows)
    y[[i]] <- ohio$resp[rows]
    Z[[i]] <- cbind(rep(1,vni[i]))
    X[[i]] <- cbind(rep(1,vni[i]), smoke[rows], age[rows], smokeage[rows])
}
n <- length(y)                              # no. of subjects
k <- dim(X[[1]])[2]                         # length of beta
p <- dim(Z[[1]])[2]                         # length of b_i 
d <- n*p + k + p*(p+1)/2
yall <- unlist(y)
Xall <- matrix(unlist(lapply(X,t)), ncol=k, byrow=TRUE)
Zall <- unlist(Z)
labels <- c('intercept','smoke','age','smoke x age','zeta')  
N <- sum(vni)
startindex <- c(0, cumsum(vni)[1:(n-1)]) + 1
endindex <- cumsum(vni)
data <- list(n=n, N=N, k=k, y=yall, X=Xall, Z=Zall, startindex=startindex, endindex=endindex)


########
# stan #
########
start_time <- Sys.time()
library(rstan)
fit <- stan(file = 'model_wheeze.stan',
            data = data, iter = 50000, chains = 1, thin=1)
#print(fit, digits = 3)


# library(bayesplot)
bayesplot::mcmc_dens(fit, regex_pars = c("beta","zeta"))
# 
la <- extract(fit, permuted = TRUE, inc_warmup=FALSE) # return a list of arrays 
end_time <- Sys.time()
print(end_time - start_time)
#write.csv(la, "SC_MCMC_t.csv")

# mcmc <- cbind(la$beta, la$zeta)

# par(mfrow=c(5,1))
# par(mar=c(1,2,1,1))
# for (i in 1:5){ plot(mcmc[,i],type='l') }
# 
# par(mfrow=c(1,5))
# for (i in 1:5){ plot(density(mcmc[,i]))}
# mcmc_mean <- apply(mcmc,2,mean)
# mcmc_sd <- apply(mcmc,2,sd)
# write.table(mcmc, file="mcmc_wheeze.txt", row.names=FALSE, col.names=FALSE)
# 
# Chain 1:  Elapsed Time: 396.793 seconds (Warm-up)
# Chain 1:                566.844 seconds (Sampling)
# Chain 1:                963.637 seconds (Total)
# Chain 1: 
# > print(fit, digits = 3)
# Inference for Stan model: model_wheeze.
# 1 chains, each with iter=50000; warmup=25000; thin=5; 
# post-warmup draws per chain=5000, total post-warmup draws=5000.
# 
#              mean se_mean     sd      2.5%       25%       50%       75%     97.5% n_eff Rhat
# beta[1]    -3.169   0.005  0.229    -3.636    -3.320    -3.164    -3.013    -2.739  2209    1
# beta[2]     0.470   0.004  0.290    -0.089     0.272     0.470     0.664     1.044  4654    1
# beta[3]    -0.220   0.001  0.087    -0.389    -0.278    -0.219    -0.162    -0.047  4834    1
# beta[4]     0.109   0.002  0.139    -0.166     0.016     0.110     0.205     0.377  4877    1
# zeta       -0.791   0.002  0.088    -0.961    -0.850    -0.792    -0.733    -0.616  1688    1
# lp__    -1275.503   0.902 36.318 -1347.090 -1299.955 -1275.065 -1251.634 -1203.764  1622    1
# 
# Samples were drawn using NUTS(diag_e) at Thu Feb 07 15:19:29 2019.
# For each parameter, n_eff is a crude measure of effective sample size,
# and Rhat is the potential scale reduction factor on split chains (at 
# convergence, Rhat=1).
