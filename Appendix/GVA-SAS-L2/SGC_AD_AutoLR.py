"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains algorithm: variational inference with centered parameterized SDGM with Pytorch automatic differentiation.

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
import scipy.stats as st
import sys

sys.path.append("Utils/")
np.set_printoptions(precision=4)
cd = lambda val: val.clone().detach()
cdg = lambda val: val.clone().detach().requires_grad_(True)
tn = torch.tensor
torch.set_default_dtype(torch.float64)


# %%
# -----------------------------------------------------------

class SGC_AD:
    '''
    Skew Decomposable Graphical Model
    '''

    def __init__(self, p, logp=[], gr_logp=[], mask=[], optimizer=[]):
        self.p = p
        self.logp, self.gr_logp = logp, gr_logp
        self.eps = 1e-8
        self.lr = 0.01
        self.lr_stepsize = 5
        self.lr_gamma = 0.5
        self.average_ELBOlog = []
        self.optimizer = optimizer

        self.prm = {'mu_': t.zeros(p, requires_grad=True),
                    # be sure to always start with 1's on diagonal for L!
                    'L': (1 * t.eye((p))).requires_grad_(True),
                    'kap_': (0 * t.ones(p)).requires_grad_(True),
                    'a_': (2 * t.ones(p)).requires_grad_(True)}

        self.Tmask = mask  # the mask that goes in is LOWER triangular!

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

    def a(self):
        ''' experimentation here could perhaps make the SDGM work better '''
        return self.prm['a_']

    # swapping the above to this seems to help
    # self.prm['L'] @ t.exp(self.prm['a_']) * t.sign(self.prm['a_'])

    def sample(self):
        ''' Generates a Single Sample '''
        self.absU, self.V = t.abs(torch.randn(self.p)), torch.randn(self.p)
        theta = self.t()
        self.lastELBO = self.ELBO(theta)
        return theta

    def sample_set(self, n):
        ''' Returns (n,p)-numpyarray  of Samples '''
        theta = np.zeros((n, self.p))
        for i in range(n):
            th = self.sample()
            theta[i, :] = th.detach().numpy()
        return theta

    def t(self):
        ''' generative function for reparametrization trick '''

        trm1 = (np.sqrt(np.pi) * (self.a() * self.absU + self.V)
                - np.sqrt(2) * self.a())
        trm2 = ((np.pi - 2) * (self.a() ** 2) + np.pi) ** (-1 / 2)
        self.Z_a = trm1 * trm2

        # --- less efficient (but equivalent way of generating below ----
        # self.Z = (1 + self.a()**2)**(-1/2) \
        #                    * (self.a()* self.absU + self.V)

        # mu_a = self.a() * ((1+self.a()**2)**(-1/2)) * np.sqrt(2/np.pi)
        # sigma_a = (1 - (2/np.pi)*(self.a()**2)/(1+self.a()**2))**(1/2)

        # self.Z_a = (self.Z - mu_a)/sigma_a

        self.x = self.kap() ** (-1 / 2) * self.Z_a

        self.Linv_x = torch.triangular_solve(self.x.reshape(-1, 1), self.prm['L'],
                                             upper=True, unitriangular=True)[0]
        theta = self.mu() + self.Linv_x.reshape(-1)
        if torch.isnan(torch.sum(theta)):
            print("error")
        return theta

    def logq(self, th):
        # need to turn these gradients off as we only want to accumulate for
        # theta! (note we use no grad so it doesnt note gradient in evaluating mu or kap)
        with t.no_grad():
            mu = self.mu().detach()
            L = self.prm['L'].detach()
            a = self.a().detach()
            kap = self.kap().detach()

        mu_a = a * ((1 + a ** 2) ** (-1 / 2)) * np.sqrt(2 / np.pi)
        sigma_a = (1 - (2 / np.pi) * (a ** 2) / (1 + a ** 2)) ** (1 / 2)

        Z_th = mu_a + sigma_a * kap ** (1 / 2) * (L @ ((th - mu)))

        log_phi = t.distributions.Normal(0, 1).log_prob(Z_th)
        log_CDF = t.log(t.distributions.Normal(0, 1).cdf(a * Z_th))

        c = self.p * np.log(2)
        logq = (c + t.sum(log_phi)
                + t.sum(log_CDF)
                + t.sum(t.log(kap ** (1 / 2)))
                + t.sum(t.log(sigma_a))
                )

        return logq

    def ELBO(self, th):
        with torch.no_grad():
            try:
                ELBO = self.logp(th) - self.logq(th)
            except:
                print("ELBO error")

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

        gr_logp = t.tensor(self.gr_logp(th).reshape(-1, 1))

        thn = th.clone().detach().requires_grad_(True)
        logq = self.logq(thn)

        # this below is the estimator of grad(logq(th))
        self.gr_entropy = t.autograd.grad(logq, thn)[0]

        r = gr_logp - self.gr_entropy.reshape(-1, 1)
        th.backward(-r.flatten())  # vector Jacobian product

        if self.prm['L'].requires_grad == True:
            self.prm['L'].grad *= self.Lmask

        self.optim.step()

