### The code below is modified from code used in the paper 
### "Gaussian variational approximation with sparse precision
### matrices" by Linda S. L. Tan and David J. Nott.
### (https://doi.org/10.1007/s11222-017-9729-7) 
### and kindly supplied by the authors of the paper.

library(rstan)

#####################
# polypharmacy data #
#####################

df <-  read.table("data_polypharm.txt", header=TRUE)
ID <- unique(df$ID)
n <- length(ID)                      # no. of subjects
vni <- rep(0,n)
X <- NULL
Z <- NULL
y <- NULL
N <- dim(df)[1]
AGEFP1 <- log(df$AGE/10)
gender_M <- df$GENDER
MHV4_1 <- rep(0,N)
MHV4_2 <- rep(0,N)
MHV4_3 <- rep(0,N)
RACE2_1 <- rep(0,N)
INPTMHV2_1 <- rep(0,N)
MHV4_1[df$MHV4==1] <- 1 
MHV4_2[df$MHV4==2] <- 1
MHV4_3[df$MHV4==3] <- 1
RACE2_1[df$RACE>0] <- 1
INPTMHV2_1[df$INPTMHV3>0] <- 1

for (i in 1:n){
  rows <- which(df$ID == ID[i])
  vni[i] <- length(rows)
  y[[i]] <- (df$POLYPHARMACY)[rows]
  Z[[i]] <- cbind(rep(1,vni[i]))
  X[[i]] <- cbind(rep(1,vni[i]), gender_M[rows], RACE2_1[rows], AGEFP1[rows],
                  MHV4_1[rows], MHV4_2[rows],MHV4_3[rows],INPTMHV2_1[rows])
}
sum(vni)                                    # total no. of observations
n <- length(y)                              # no. of subjects
k <- dim(X[[1]])[2]                         # length of beta
p <- dim(Z[[1]])[2]                         # length of b_i 
d <- n*p + k + p*(p+1)/2
yall <- unlist(y)
Xall <- matrix(unlist(lapply(X,t)), ncol=k, byrow=TRUE)
Zall <- unlist(Z)
N <- as.integer(sum(vni))    
startindex <- as.integer(c(0, cumsum(vni)[1:(n-1)]) + 1)
endindex <- as.integer(cumsum(vni))
data <- list(n=n, N=N, k=k, y=yall, X=Xall, Z=Zall, startindex=startindex, endindex=endindex)
#saveRDS(data, file = "Polypharmacy_data.rds")

# stan #
start_time <- Sys.time()
fit <- stan(file = 'model_polypharm_t.stan',
            data = data, iter = 50000, chains = 1, thin=5)

end_time <- Sys.time()
print(end_time - start_time)


#print(fit, digits = 3)
la <- extract(fit, permuted = TRUE) # return a list of arrays 
#write.csv(la, "Poly_MCMC_full_t.csv")