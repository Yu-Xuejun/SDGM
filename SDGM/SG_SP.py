"""
Sparse Implementation of (Reparametrized) Centered
Skew Gaussian Decomposable Graphical Model


ALLOWS FOR IMPORTANCE WEIGHTED BOUND

Rob Salomone, 2020
"""
#%%
# ------ Imports ---------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as td
import torch as t
import torch.optim as optim
import sys
sys.path.append("Utils/")
import scipy.special as sp
np.set_printoptions(precision=4)
import torch_optimizer as topt
import torch
import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)

cd = lambda val: val.clone().detach() 
cdg = lambda val: val.clone().detach().requires_grad_(True)

tn = torch.tensor
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve_triangular
import scipy.stats as st
import scipy.sparse as sps

import scipy.stats as st

#%%
torch.set_default_dtype(torch.float64)

class SG_SP:
   ''' mask should be a sparse numpy array in COO format'''  
    
   def __init__(self, p, logp = [], gr_logp = [], eps=1e-8, mask = [], up_a = True, fix_global_alpha_idx = [], optimizer = [], K = int(1)):
        self.p =  p 
        self.logp, self.gr_logp = logp, gr_logp
        self.eps = eps
        self.lr=0.01
        self.optimizer = optimizer
        self.ELBOlog = []
        self.average_ELBOlog = []
        
        self.Lmask = t.tensor(t.triu(mask.T, diagonal = 1))
        self.Lmask_ = coo_matrix(self.Lmask.numpy())
        assert(K == int(K), 'K must be an integer')
        self.K = int(K)
        
        self.L_ = self.Lmask_ * 0 

        self.prm = {'mu': (0*t.ones(p)),
                    'L': t.from_numpy(self.L_.data), 
                    '_kap': (0*t.ones(p)),
                    'a': (0*t.ones(p))}

        self.up_a = up_a # set to false to not have skewness!
        self.fix_global_alpha_idx = fix_global_alpha_idx
        self.mu = self.prm['mu'].numpy()
        self.a = self.prm['a'].numpy()
        self._kap = self.prm['_kap'].numpy()



   def sample_set(self, n):
        ''' Returns (n,p)-numpyarray  of Samples ''' 
        theta = np.zeros((n, self.p))

        for i in range(n):
            th = self.sample()
            theta[i,:] = th

        return theta

    
   def sample(self):
        self.absU, self.V = t.abs(torch.randn(self.p)), torch.randn(self.p)
        self.absU, self.V = self.absU.numpy(), self.V.numpy()
        
        theta = self.h()
        return theta
      

   def spOP(self,u,v,M): 
        ''' 
        sparse Outer Product: u @ v.T w/ zeros matching sparsity pattern of M
        ''' 
        _u, _v = u[M.row], v[M.col]    
        vals = _u  * _v
        
        M_out = coo_matrix((vals,(M.row, M.col)),shape=(self.p,self.p))
        
        return M_out
   
    
   def h(self):
        self.L = self.L_ + sps.eye(self.p)
       
        self.kap = np.exp(self._kap)
        
        self.Z_a =  (1 + self.a**2)**(-1/2) \
                        * (self.a* self.absU + self.V)
                        
        self.X = self.kap**(-1/2) * self.Z_a                

        self.Y = spsolve_triangular(self.L, self.X,
                                      lower=False, unit_diagonal=True)
        
        
        theta = self.mu + self.Y.reshape(-1)
        
        return theta
 
    
   def logq_(self, th):
         Z_th = self.kap**(1/2) * (self.L @ ((th - self.mu)))
         
         log_phi = st.norm.logpdf(Z_th)
         log_CDF = st.norm.logcdf(self.a * Z_th)
         
         c = self.p * np.log(2)
         self.last_logq_ = ( c + np.sum(log_phi) +
                                 np.sum(log_CDF) + 
                                 np.sum(np.log(self.kap**(1/2))))
           
         gr1 = - Z_th
         log_PDF = st.norm.logpdf(self.a * Z_th)
         gr2 = self.a * np.exp(log_PDF - log_CDF)
         self.gr_entropy_ =  self.L.T @ (self.kap**(1/2) * (gr1 + gr2))
         

          
   def full_train(self, num_steps, show = int(500), N=1):

         if self.optimizer == "Adam":
             self.optim = optim.Adam(self.prm.values(),lr=self.lr)
         else:
             self.optim = optim.Adadelta(self.prm.values(), rho=0.9, eps=self.eps)
   
         mn_IWO = 0
        
         for i in range(1,num_steps+1):
             if i < 0:
                 self.update_alpha = False
             else:
                 self.update_alpha = True
             self.optim.zero_grad()
             self.train_step() 
             mn_IWO += self.last_IWO
             self.ELBOlog.append(float(self.last_IWO))
             if i%show == 0 and i>0:
                 print(i, float(mn_IWO/show))
                 self.average_ELBOlog.append(float(mn_IWO/show))
                 mn_IWO = 0

   def get_VJPs(self, r, j):
         r = -r.flatten() # maximize (for PyTorch Optimizer)

         self.slv = spsolve_triangular(self.L.T, r, lower=True,
                                        unit_diagonal=True)
  
         D_a =   self.kap**(-1/2) * (1+self.a**2)**(-3/2) * (self.absU - self.a * self.V) 
     
         self.mu_grad[j,:] = t.from_numpy(r.flatten())
         self.kap_grad[j,:] = t.from_numpy(-0.5 * self.X * self.slv)
         if self.update_alpha:
            self.a_grad[j,:] = t.from_numpy(D_a * self.slv) * self.up_a
         self.L_grad[j,:] = t.from_numpy(-self.spOP(self.slv, self.Y, self.Lmask_).data) 
    
   def train_step(self):    
        self.mu_grad = t.zeros(self.K, self.prm['mu'].shape[0])
        self.kap_grad = t.zeros(self.K, self.prm['_kap'].shape[0])
        self.a_grad = t.zeros(self.K, self.prm['a'].shape[0])
        self.L_grad = t.zeros(self.K, self.prm['L'].shape[0])
        
        log_w = t.zeros(self.K)
      
        for j in range(self.K):
            th = self.sample()
            self.logq_(th)
            
            logq = self.last_logq_
            logp = self.logp(th)
            log_w[j] = logp - logq
            #if log_w[j]>0:
            #    print("error")
            
            gr_logp = self.gr_logp(th).reshape(-1)         
            r = gr_logp - self.gr_entropy_     
            self.get_VJPs(r, j)

        self.last_IWO = t.logsumexp(log_w, dim=0) - np.log(self.K)
        # if self.last_IWO>0:
        #     print("logp",logp)
        #     print("logq",logq)
        norm_w = t.exp(log_w - t.logsumexp(log_w, dim=0)) # normalized weights
        w_tilde = norm_w ** 2  # weights for IWO gradient

        
        for j in range(self.K):
            self.mu_grad[j,:] = w_tilde[j] * self.mu_grad[j,:] 
            self.kap_grad[j,:] =  w_tilde[j] * self.kap_grad[j,:]
            if self.update_alpha:
                self.a_grad[j,:] =  w_tilde[j] * self.a_grad[j,:]
            self.L_grad[j,:] = w_tilde[j] * self.L_grad[j,:]
         
        self.prm['mu'].grad = self.mu_grad.sum(axis=0)
        self.prm['_kap'].grad = self.kap_grad.sum(axis=0)
        if self.update_alpha:
            self.prm['a'].grad = self.a_grad.sum(axis=0)
            self.prm['a'].grad[self.fix_global_alpha_idx] = 0
        self.prm['L'].grad = self.L_grad.sum(axis=0)
        self.optim.step()
     
        