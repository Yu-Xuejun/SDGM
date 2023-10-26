"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains algorithm: variational inference with centered SDGM with copula methods.

Rob Salomone, Yu Xuejun, Sept 2022.
"""
# %%
# ------ Imports ---------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as td
import torch as t
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import sys

sys.path.append("Utils/")
import scipy.special as sp

np.set_printoptions(precision=4)
import scipy.stats as st

cd = lambda val: val.clone().detach()
cdg = lambda val: val.clone().detach().requires_grad_(True)
import torch_optimizer as topt

tn = torch.tensor



def asinh(x):
    return t.log1p(x * (1 + x / (t.sqrt(x ** 2 + 1) + 1)))


import torch_optimizer as topt

# %%
# ---- Import Custom Modules / Useful Functions -------------
torch.set_default_dtype(torch.float64)


class SGGM:
    ''' mask should be a sparse numpy array in COO format'''

    def __init__(self, p, logp=[], gr_logp=[], mask=[], up_a=True, up_eps=True, up_kurtosis=False, centered=True,
                 optimizer=[]):
        self.p = p
        self.logp, self.gr_logp = logp, gr_logp
        self.eps = 1e-8
        L = t.eye(p)
        self.average_ELBOlog = []
        # self.skew = False
        self.lr = 0.01
        self.lr_stepsize = 5
        self.lr_gamma = 0.1
        self.optimizer = optimizer

        self.prm = {'mu': t.zeros(p, requires_grad=True),  # (p,)
                    'L': L.requires_grad_(True),  # (p, r_cov)
                    '_sigma': (-3 * t.ones(p)).requires_grad_(True),
                    'a': t.zeros(p, requires_grad=up_a),
                    'eps': t.zeros(p, requires_grad=up_eps),
                    'delta': t.ones(p, requires_grad=up_kurtosis)
                    }

        self.mask = t.triu(mask.T, diagonal=1)
        self.centered = centered

    def sample_set(self, n):
        ''' Returns (n,p)-numpyarray  of Samples '''
        theta = np.zeros((n, self.p))
        for i in range(n):
            th = self.sample()
            theta[i, :] = th.detach().numpy()
        return theta

    def sample(self):
        self.V = torch.randn(self.p)
        self.absU = t.abs(torch.randn(self.p))
        theta = self.h()

        self.logq(theta)
        self.last_logp = self.logp(theta)

        self.lastELBO = self.ELBO(theta)
        return theta

    def h(self):
        if self.centered:
            trm1 = (np.sqrt(np.pi) * (self.prm['a'] * self.absU + self.V)
                    - np.sqrt(2) * self.prm['a'])
            trm2 = ((np.pi - 2) * (self.prm['a'] ** 2) + np.pi) ** (-1 / 2)
            self.Z_a = trm1 * trm2
        else:
            self.Z_a = (1 + self.prm['a'] ** 2) ** (-1 / 2) \
                       * (self.prm['a'] * self.absU + self.V)

        self.X = torch.triangular_solve(self.Z_a.reshape(-1, 1), self.prm['L'], upper=True, unitriangular=True)[
            0].reshape(-1)

        self.Y = self.t_inv()

        self.sigma = t.exp(self.prm['_sigma'])
        theta = self.prm['mu'] + self.sigma * self.Y.reshape(-1)

        return theta

    def t_inv(self):
        return t.sinh(self.prm['delta'] ** (-1) * (asinh(self.X) + self.prm['eps']))

    def t(self, Y, eps, delta):
        self.c2 = eps - delta * asinh(Y)
        return t.sinh(-self.c2)

    def t_idash(self, X, eps, delta):
        return t.cosh(delta ** (-1) * (asinh(X) + eps)) / (t.sqrt(1 + X ** 2) * delta)

    def logq(self, th):

        # ------- stop gradient for pathwise ------
        mu = self.prm['mu'].detach()
        L = self.prm['L'].detach()
        a = self.prm['a'].detach()
        eps = self.prm['eps'].detach()
        delta = self.prm['delta'].detach()
        sigma = t.exp(self.prm['_sigma'].detach())
        # # -------------------------------------------

        thn = th.detach().clone().reshape(-1).requires_grad_(True)

        if self.centered:
            mu_a = a * ((1 + a ** 2) ** (-1 / 2)) * np.sqrt(2 / np.pi)
            sigma_a = (1 - (2 / np.pi) * (a ** 2) / (1 + a ** 2)) ** (1 / 2)
        else:
            mu_a = 0
            sigma_a = 0

        y = (thn - mu) / sigma
        t_y = self.t(y, eps, delta)

        Z_a = mu_a + sigma_a * (L @ t_y)



        logpdfs = td.Normal(loc=0, scale=1).log_prob(Z_a)
        logcdfs_ = t.log(td.Normal(loc=0, scale=1).cdf(a * Z_a))
        # logcdfs_ = t.tensor(st.norm.logcdf((a * Z_a).detach().numpy()))

        tdash = self.t_idash(t_y, eps, delta)

        C = self.p * np.log(2)

        self.last_logq = C + t.sum(logcdfs_) + t.sum(t.log(sigma_a)) + t.sum(logpdfs) - t.sum(t.log(tdash)) - t.sum(
            t.log(sigma))

        self.last_logq.backward()
        self.gr_entropy = thn.grad
        return self.last_logq

    def ELBO(self, th):
        with torch.no_grad():
            ELBO = self.last_logp - self.last_logq
        return ELBO

    def full_train(self, num_steps, show=int(500), N=1):
        if self.optimizer == "Adam":
            self.optim = optim.Adam(self.prm.values(), lr=self.lr)
        else:
            self.optim = optim.Adadelta(self.prm.values(), rho=0.9, eps=self.eps)

        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=self.lr_stepsize, gamma=self.lr_gamma)
        mn_ELBO = 0

        for epoch in range(1,int(num_steps/1000)+1):
            for i in range(1, 1000 + 1):
                self.optim.zero_grad()
                self.train_step()
                mn_ELBO += self.lastELBO
                if i % show == 0 and i > 0:
                    print(i, float(mn_ELBO / show))
                    self.average_ELBOlog.append(float(mn_ELBO / show))
                    mn_ELBO = 0
            before_lr = self.optim.param_groups[0]["lr"]
            self.scheduler.step()
            after_lr = self.optim.param_groups[0]["lr"]
            print("Epoch %d: lr %.12f -> %.12f" % (epoch, before_lr, after_lr))

    def train_step(self):

        th = self.sample()
        self.logq(th)

        r = self.gr_logp(th).reshape(-1, 1) - self.gr_entropy.reshape(-1, 1)
        th.backward(-r.flatten())

        if self.prm['L'].requires_grad == True:
            self.prm['L'].grad = self.prm['L'].grad * self.mask

        self.optim.step()


