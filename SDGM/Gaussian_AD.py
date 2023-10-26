"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains algorithm: variational inference with Gaussian Graphical Model with Pytorch automatic differentiation.

Note that this is equivalent to SDGM with alpha set to the zero vector. This file is only to remove redundant code to benchmark
speed difference.
Note that in this code, L is an upper triangular matrix.

Rob Salomone, Yu Xuejun, Sept 2022.
"""
#%%
# ------ Imports ---------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.distributions as td
import torch as t
import torch.optim as optim
import scipy.stats as st     
import sys
sys.path.append("Utils/")
np.set_printoptions(precision=4)
cd = lambda val: val.clone().detach() 
cdg = lambda val: val.clone().detach().requires_grad_(True)
tn = torch.tensor
torch.set_default_dtype(torch.float64) 
#%%
# -----------------------------------------------------------

class SG_AD:
   '''
   Skew Decomposable Graphical Model
   '''
    
   def __init__(self, p, logp = [], gr_logp = [], mask = [], up_a = True, optimizer = []):
        self.p =  p 
        self.logp, self.gr_logp = logp, gr_logp
        self.eps = 1e-8
        self.lr = 0.01
        self.average_ELBOlog = []
        self.optimizer = optimizer

        self.prm = {'mu_': t.zeros(p, requires_grad=True),
                     # be sure to always start with 1's on diagonal for L!
                    'L': (1*t.eye((p))).requires_grad_(True),
                    'kap_': (0*t.ones(p)).requires_grad_(True)}
      
        self.Tmask = mask # the mask that goes in is LOWER triangular!
        
        # this makes it upper triangular and makes sure the diagonal is 
        # never updated (it needs to stay 1's, so it makes diagonal zero)
        self.Lmask = t.triu(mask.T, diagonal=1)   
   
    
   # note that one can in principle modify kap and mu to give a 
   # 'centered' parametrization
   def kap(self):
       ''' 
       this is a special function that allows you to easioy change how
       kappa is parametrized. All you need to do is change how you turn 
       the elements of self.prm into kap and it does the gradients itself!
        
       note that the elements of the variational parameters dictionary 
       are written with a "_" at the end, this is my own shorthand to remind
       myself that they are going to be further fed through a function. 
        
       e.g., you use kap_ to get kap 
       ''' 
       
       return t.exp(self.prm['kap_'])   

   def mu(self):
       return self.prm['mu_'] 

   
    # swapping the above to this seems to help
   #self.prm['L'] @ t.exp(self.prm['a_']) * t.sign(self.prm['a_'])
              
   def sample(self):    
       ''' Generates a Single Sample '''
       self.V = torch.randn(self.p) 
       theta = self.t()
       self.lastELBO = self.ELBO(theta)
       return theta
      
   def sample_set(self, n):
        ''' Returns (n,p)-numpyarray  of Samples ''' 
        theta = np.zeros((n, self.p))
        for i in range(n):
            th = self.sample()
            theta[i,:] = th.detach().numpy()
        return theta

   def t(self):
       ''' generative function for reparametrization trick '''
       self.x = self.kap() ** (-1 / 2) * self.V
       self.Linv_x = torch.triangular_solve(self.x.reshape(-1,1), self.prm['L'],
                                            upper=True, unitriangular=True)[0]
       theta = self.mu() + self.Linv_x.reshape(-1)
        
       return theta

   def logq(self, th):
       # need to turn these gradients off as we only want to accumulate for 
       # theta! (note we use no grad so it doesnt note gradient in evaluating mu or kap)
        with t.no_grad():
            mu = self.mu().detach()
            L = self.prm['L'].detach()
            kap = self.kap().detach() 
        
        # Q = L.T @ t.diag(kap) @ L
        
        th_min_mu = th - mu
        
        # logphi = (self.p/2)*np.log(2/np.pi) + 0.5* t.log(t.prod(kap)) \
        #                 - 0.5 * th_min_mu.T @ Q @ th_min_mu

        Z_th = kap ** (1 / 2) * (L @ th_min_mu)
        logphi_test = td.Normal(0,1).log_prob(Z_th)

        # logq = t.sum(logphi_test)
        # c = self.p * np.log(2)
        logq = (t.sum(logphi_test)
                + t.sum(t.log(kap ** (1 / 2)))
                )
        # logq_original = logphi + t.sum(logcdfs)
        # print(logq - logq_original)
         
        return logq
    
   def ELBO(self, th):
        with torch.no_grad(): 
            ELBO = self.logp(th) -self.logq(th)
        
        return ELBO
            
   def full_train(self, num_steps, show = int(500), N=1):

        if self.optimizer == "Adam":
            self.optim = optim.Adam(self.prm.values(), lr=self.lr)
        else:
            self.optim = optim.Adadelta(self.prm.values(), rho=0.9, eps=self.eps)

        mn_ELBO = 0
        
        for i in range(1,num_steps+1):
            self.optim.zero_grad()
            self.train_step() 
            mn_ELBO += self.lastELBO
            if i%show == 0 and i>0:
                print(i, float(mn_ELBO/show))
                self.average_ELBOlog.append(float(mn_ELBO / show))
                mn_ELBO = 0
                  
   def train_step(self, save = True):    
        th = self.sample()

        gr_logp = t.tensor(self.gr_logp(th).reshape(-1,1))

        thn = th.clone().detach().requires_grad_(True)
        logq = self.logq(thn)
        
        # this below is the estimator of grad(logq(th))
        self.gr_entropy = t.autograd.grad(logq, thn)[0]

        r = gr_logp - self.gr_entropy.reshape(-1,1)
        th.backward(-r.flatten()) # vector Jacobian product
        
        # next line zeroes out elements of the gradient to match
        # desired sparsity structure
        
        if self.prm['L'].requires_grad == True: 
            self.prm['L'].grad *= self.Lmask 
               
        # this makes the optimizer take a step, and zeros out the gradients
        self.optim.step()
     
        
     