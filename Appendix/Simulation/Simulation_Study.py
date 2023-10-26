# %%
import torch
from torch.distributions import Bernoulli
from sklearn.metrics import precision_score, recall_score, accuracy_score

def prediction_evaluation(y_test, X, beta, z):
    '''
    beta: (n_samples, n_fixed_effects) torch.Tensor 
    z: (n_samples, n_individuals) torch.Tensor
        z_ik is the i-th posterior sample for the k-th individual
    X: (n_individuals, n_fixed_effects) torch.Tensor -
                Design matrix for fixed effects
    '''
    
    n_samples = beta.shape[0] 
    
    assert beta.shape[0] == z.shape[0] 
    
    pll = torch.zeros(n_samples)
    acc = torch.zeros(n_samples)
    acc2 = torch.zeros(n_samples)

    for k in range(n_samples):
        # loop is over samples from posterior 
        # (either MCMC or from variational approximation)
        # compute logit(p) for each test point 
        # and prediction
        
        # beta[k,:].shape = (n_fixed effects) 
        # so we reshape below to get a column vector
        # of shape (n_fixed_effects, 1) 
        
        logits = X @ beta[k,:].reshape(-1,1) + z[k,:].reshape(-1,1)
        logits = logits.reshape(1,-1)
        y_test = y_test.reshape(1,-1)

        logits = torch.tensor(logits,dtype=torch.float64)
        y_test = torch.tensor(y_test,dtype=torch.float64)

        plls = Bernoulli(logits=logits).log_prob(y_test)
        
        # mean across data observations 
        pll[k] = torch.mean(plls)
        
        y_pred = (torch.sigmoid(logits) > 0.5) * 1.
        y_pred = torch.tensor(y_pred,dtype=torch.float64)
        y_test = torch.squeeze(y_test)
        y_pred = torch.squeeze(y_pred)
        acc[k] = accuracy_score(y_test, y_pred)
        acc2[k] = torch.sum(torch.eq(y_test, y_pred))/y_pred.__len__()
    
    # compute mean predictive log-likelihood and accuracy over 
    # all posterior samples 
    return torch.mean(pll).item(), torch.mean(acc).item(), torch.mean(acc2).item()

# %%
"""
# Tests 
"""

# %%
# Test 1  
# y_test = (torch.rand(10) < 0.5) * 1.
#
# logits = torch.randn(10)
# y_pred = (torch.sigmoid(logits) > 0.5) * 1.
#
# torch.mean(Bernoulli(logits=logits).log_prob(y_test))
# print(accuracy_score(y_test, y_pred))
# print(Bernoulli(logits=logits).log_prob(y_test))
# print(y_test)
# a = torch.rand(10) < 0.5
# print(a)
# print(a * 1.)