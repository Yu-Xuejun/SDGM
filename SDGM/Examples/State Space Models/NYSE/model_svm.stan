### The code below is modified from code used in the paper 
### "Conditionally structured variational Gaussian approximation 
### with importance weights" by Linda S. L. Tan, Aishwarya Bhaskaran 
### and David J. Nott. (https://doi.org/10.1007/s11222-020-09944-8) 
### and kindly supplied by the authors of the paper.
 data {
   int<lower=0> n; // number of subjects
   vector[n] y; // responses
 }
 parameters {
   vector[n] x;
      real alpha;
 real kappa;
   real psi;
}
transformed parameters{
   real phi;
  phi = 1/(exp(-psi)+1);
 }
model {
  target += normal_lpdf(alpha | 0, sqrt(10) );
  target += normal_lpdf(kappa |0, sqrt(10) );
  target += normal_lpdf(psi | 0, sqrt(10));

  target += normal_lpdf(x[1]| 0, 1/sqrt(1-phi^2));

  for (t in 2:n){
    target += normal_lpdf(x[t]| phi * x[t-1], 1);
 }

  for (t in 1:n){
    target += normal_lpdf(y[t]|0, exp(kappa/2 + log(exp(alpha)+1) * x[t]/2));
   }

}

