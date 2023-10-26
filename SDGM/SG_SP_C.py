"""
Sparse Implementation of (Reparametrized) Centered
Skew Gaussian Decomposable Graphical Model

Rob Salomone, 2020
"""
#%%
# ------ Imports ---------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.distributions as td
import torch as t
import torch.optim as optim
import sys
from tqdm import tqdm
sys.path.append("Utils/")
import scipy.special as sp
np.set_printoptions(precision=4)
import torch_optimizer as topt

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
torch.set_default_dtype(torch.float64)

class SG_SP_C:
   ''' mask should be a sparse numpy array in COO format'''  
    
   def __init__(self, p, logp = [], gr_logp = [], eps=1e-8, mask = [],fix_global_alpha_idx = [], optimizer=[], K=1):
        self.p =  p 
        self.logp, self.gr_logp = logp, gr_logp
        self.eps = eps
        self.lr = 0.01
        self.optimizer = optimizer
        self.K = K
        self.ELBOlog = []
        self.average_ELBOlog = []

        mask = t.tensor(mask)
        self.Lmask = t.triu(mask.T, diagonal = 1)
        self.Lmask_ = coo_matrix(self.Lmask.numpy())
        self.L_ = self.Lmask_ * 0
        self.prm = {'mu': (0*t.ones(p)),
                    'L': t.from_numpy(self.L_.data), 
                    '_kap': (0*t.ones(p)),
                    'a': (5*t.ones(p))
                    }
        
        self.up = {'mu': 1,
                    'L': 1, 
                    '_kap': 1,
                    'a': 1}

        self.fix_global_alpha_idx = fix_global_alpha_idx
        self.prm['a'][self.fix_global_alpha_idx] = 0

        self.mu = self.prm['mu'].numpy()
        self.a = self.prm['a'].numpy()
        self._kap = self.prm['_kap'].numpy()

   def sample_set(self, n,sinh_trans = False,trans_idx=[],delta=0.5):
        ''' Returns (n,p)-numpyarray  of Samples ''' 
        theta = np.zeros((n, self.p))

        def sinh_arcsinh(x):
            epsilon = 0
            t = np.sinh(delta * (np.arcsinh(x)) - epsilon)
            return t

        for i in range(n):
            th = self.sample()
            if sinh_trans:
                th[trans_idx] = sinh_arcsinh(th[trans_idx])

            theta[i,:] = th
        return theta

    
   def sample(self):    
        self.absU, self.V = t.abs(torch.randn(self.p)), torch.randn(self.p)
        self.absU, self.V = self.absU.numpy(), self.V.numpy()
        
        theta = self.h()
        
        self.logq_(theta)
        self.lastELBO = self.ELBO(theta)
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

        # YXJ0630
        # self.Z_a_ =  (1 + self.a**2)**(-1/2) \
        #                 * (self.a* self.absU + self.V)
        # mu_a = self.a * ((1+self.a**2)**(-1/2)) * np.sqrt(2/np.pi)
        # sigma_a = (1 - (2/np.pi)*(self.a**2)/(1+self.a**2))**(1/2)
        # self.Z_a = (self.Z_a_ - mu_a)/sigma_a
        # self.X = self.kap ** (-1 / 2) * sigma_a * self.Z_a
        # self.Y = spsolve_triangular(self.L, self.X,
        #                               lower=False, unit_diagonal=True)
        # theta = self.mu + self.Y.reshape(-1)

        self.L = self.L_ + sps.eye(self.p)
        trm1 = (np.sqrt(np.pi) * (self.a * self.absU + self.V)
                      - np.sqrt(2)*self.a)
        trm2 =((np.pi -2)*(self.a**2) + np.pi)**(-1/2)
        self.Z_a = trm1 * trm2

        # ---- this is same as above but slightly less efficient ------ 
        # self.Z_a_ =  (1 + self.a**2)**(-1/2) \
        #                 * (self.a* self.absU + self.V)
        # mu_a = self.a * ((1+self.a**2)**(-1/2)) * np.sqrt(2/np.pi)
        # sigma_a = (1 - (2/np.pi)*(self.a**2)/(1+self.a**2))**(1/2)
        # self.Z_a = (self.Z_a_ - mu_a)/sigma_a
        # -------------------------------------------------------------
        
              
        self.X = self.kap**(-1/2) * self.Z_a

        self.Y = spsolve_triangular(self.L, self.X,
                                      lower=False, unit_diagonal=True)
        

        theta = self.mu + self.Y.reshape(-1)
        if np.isnan(np.sum(theta)):
            print("error")

        return theta
 
    
   def logq_(self, th):
         
         mu_a = self.a * ((1+self.a**2)**(-1/2)) * np.sqrt(2/np.pi)
         sigma_a = (1 - (2/np.pi)*(self.a**2)/(1+self.a**2))**(1/2)

         Z_th = mu_a + sigma_a * self.kap**(1/2) * (self.L @ ((th - self.mu)))
         # Z_th = mu_a + self.kap ** (1 / 2) * (self.L @ ((th - self.mu))) #YXJ0630


         log_phi = st.norm.logpdf(Z_th)
         log_CDF = st.norm.logcdf(self.a * Z_th)
         
         c = self.p * np.log(2)
         self.last_logq_ = ( c + np.sum(log_phi) 
                               + np.sum(log_CDF) + 
                               + np.sum(np.log(self.kap**(1/2))) 
                               + np.sum(np.log(sigma_a))
                            )
         
         
         gr1 = - Z_th
         log_PDF = st.norm.logpdf(self.a * Z_th)
         gr2 = self.a * np.exp(log_PDF - log_CDF)
         
         self.gr_entropy_ =  self.L.T @ (self.kap**(1/2) * sigma_a * (gr1 + gr2))
       
  
    
 
   def ELBO(self, th):
         with torch.no_grad(): 
             ELBO = self.logp(th) - self.last_logq_
        
         return ELBO
            
   def full_train(self, num_steps, show = int(500), N=1):
         if self.optimizer == "Adam":
             self.optim = optim.Adam(self.prm.values(),lr=self.lr)
         else:
             self.optim = optim.Adadelta(self.prm.values(), rho=0.9, eps=self.eps)

         mn_ELBO = 0
        
         for i in tqdm(range(1,num_steps+1)):
             if i < 0:
                 self.update_alpha = False
             else:
                 self.update_alpha = True
             self.optim.zero_grad()
             self.train_step() 
             mn_ELBO += self.lastELBO
             self.ELBOlog.append(self.lastELBO)
             if i%show == 0 and i>0:
                 print(i, float(mn_ELBO/show))
                 self.average_ELBOlog.append(float(mn_ELBO / show))
                 mn_ELBO = 0
    
   # def get_VJPs(self, r):
   #       r = -r.flatten() # maximize (for PyTorch Optimizer)

   #       self.slv = spsolve_triangular(self.L.T, r, lower=True,
   #                                      unit_diagonal=True)
  
   #       # diag(dZ_d_a)
   #       D_a = -np.sqrt(np.pi) * ((np.pi - 2)*(self.a) * self.V 
   #                     - np.pi * self.absU + np.sqrt(2*np.pi))\
   #                  * ((np.pi-2)*(self.a**2) + np.pi)**(-3/2)  
                    
                    
   #       self.prm['mu'].grad = t.from_numpy(r.flatten()) * self.up['mu']
   #       self.prm['_kap'].grad = t.from_numpy(-0.5 * self.X * self.slv) * self.up['_kap']
   #       self.kap = np.exp(self._kap)
   #       self.prm['a'].grad = t.from_numpy(self.kap**(-1/2) * D_a * self.slv)* self.up['a']
   #       self.prm['L'].grad = t.from_numpy(-self.spOP(self.slv, self.Y, self.Lmask_).data) * self.up['L'] 
    
   # def train_step(self):    
   #       th = self.sample()

   #       gr_logp = self.gr_logp(th).reshape(-1)         
   #       r = gr_logp - self.gr_entropy_     
   #       self.get_VJPs(r)

   #       self.optim.step()
      
   def get_VJPs(self, r, j):
          self.kap = np.exp(self._kap)
          r = -r.flatten() # maximize (for PyTorch Optimizer)

          self.slv = spsolve_triangular(self.L.T, r, lower=True,
                                         unit_diagonal=True) # L^-T * r
  
          # diag(dZ_d_a)
          D_a = -np.sqrt(np.pi) * ((np.pi - 2)*(self.a) * self.V 
                        - np.pi * self.absU + np.sqrt(2*np.pi))\
                     * ((np.pi-2)*(self.a**2) + np.pi)**(-3/2)  
                             
          self.mu_grad[j,:] = t.from_numpy(r.flatten()) * self.up['mu']
          self.kap_grad[j,:] = t.from_numpy(-0.5 * self.X * self.slv) * self.up['_kap']
          if self.update_alpha:
              self.a_grad[j,:] = t.from_numpy(self.kap**(-1/2) * D_a * self.slv)* self.up['a']
          self.L_grad[j,:] = t.from_numpy(-self.spOP(self.slv, self.Y, self.Lmask_).data) * self.up['L']
    
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
            
              gr_logp = self.gr_logp(th).reshape(-1)         
              r = gr_logp - self.gr_entropy_     
              self.get_VJPs(r, j)

          self.last_IWO = t.logsumexp(log_w, dim=0) - np.log(self.K)
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
     

