"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains collection of numerical examples.

Rob Salomone, Yu Xuejun, Sept 2022.
"""

import numpy as np
import scipy.special as sp
from scipy.linalg import solve_triangular
import scipy.stats as st
import os
import torch.distributions as td
import sys
import pickle
import scipy
import torch 
import pandas as pd 
import torch as t 
sys.path.append("Utils")

from Rtools import rds2dict

from STANtools import Stan2Py, cVec

import numpy as np 
import scipy.stats as st

def init_Epilepsy_Slope():
    root =  "".join([os.getcwd(), "/Examples/Linear Mixed Effects/Epilepsy/"])
    STANfile = "".join([root, "model_epilepsy_slope.stan"])
    datfile = "".join([root,"EpilepsySlopeData.rds"])

    data = rds2dict(datfile) 
    EX = Stan2Py(STANfile, data, name = "EpilepsySlope")
    EX.G = 9 # number of globals
    EX.L = 59 # number of individuals

    A = scipy.linalg.block_diag(*([np.array([[1,0],[1,1]])]*EX.L))
    B = np.zeros((EX.L*2,EX.G))
    C = np.ones((EX.G,EX.G+(EX.L*2)))
    EX.Tmask = np.tril(np.block([[A,B],[C]]))
    EX.Tmask = torch.tensor(EX.Tmask)

    MCMCfile = "".join([root, 'EpilepsyMCMC.csv'])
    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMC = np.array(EX.MCMCdf)[:,1:]
    EX.MCMC = np.array(EX.MCMC)[:, :-1]
    return EX

def init_Epilepsy_Slope_t():
    root =  "".join([os.getcwd(), "/Examples/Linear Mixed Effects/Epilepsy/"])
    STANfile = "".join([root, "model_epilepsy_slope_t.stan"])
    datfile = "".join([root,"EpilepsySlopeData_t.rds"])

    data = rds2dict(datfile)
    EX = Stan2Py(STANfile, data, name = "EpilepsySlope_t")
    G = 9 # number of globals
    L = 59 # number of individuals

    A = scipy.linalg.block_diag(*([np.array([[1,0],[1,1]])]*L))
    B = np.zeros((L*2,G))
    C = np.ones((G,G+(L*2)))
    EX.Tmask = np.tril(np.block([[A,B],[C]]))
    EX.Tmask = torch.tensor(EX.Tmask)

    MCMCfile = "".join([root, 'EpilepsyMCMC_t.csv'])
    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMC = np.array(EX.MCMCdf)[:,1:]
    return EX


def init_SixCities():
    root = "".join([os.getcwd(), "/Examples/Linear Mixed Effects/SixCities/"])
    STANfile = "".join([root, "model_wheeze.stan"])
    datfile = "".join([root, "wheezedata.rds"])     
    data = rds2dict(datfile)
    EX = Stan2Py(STANfile, data, name = "SixCities")
    MCMCfile = datfile = "".join([root, "SC_MCMC_FULL.csv"])
    
    Tmask = np.eye(542)
    Tmask[537:,:] = np.ones((5,542))
    EX.Tmask = torch.tensor(np.tril(Tmask))
    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMC = np.array(EX.MCMCdf)[:,1:]


    ids_rand = list(range(537))
    a = np.where(np.mean(EX.MCMC[:, ids_rand], 0) < 0)
    EX.index_MCMC = a[0]
    EX.n_MCMC = len(EX.index_MCMC)
    b = list(range(542))
    EX.index_VB = [x for x in b if (x not in EX.index_MCMC)]

    EX.G = 5
    EX.L = 537

    return EX


def init_SixCities_t():
    root = "".join([os.getcwd(), "/Examples/Linear Mixed Effects/SixCities/"])
    STANfile = "".join([root, "model_wheeze_t.stan"])
    datfile = "".join([root, "wheezedata.rds"])
    data = rds2dict(datfile)
    EX = Stan2Py(STANfile, data, name="SixCities_t")
    MCMCfile = datfile = "".join([root, "SC_MCMC_t.csv"])

    Tmask = np.eye(542)
    Tmask[537:, :] = np.ones((5, 542))
    EX.Tmask = torch.tensor(np.tril(Tmask))
    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMC = np.array(EX.MCMCdf)[:, 1:]

    EX.G = 5
    EX.L = 537

    return EX


def init_Polypharmacy():
    root = "".join([os.getcwd(), "/Examples/Linear Mixed Effects/Polypharmacy/"])
    STANfile = "".join([root, "model_polypharm.stan"])
    datfile = "".join([root, "Polypharmacy_data.rds"])
    MCMCfile = "".join([root, "Poly_MCMC_full.csv"])
    data = rds2dict(datfile)
    EX = Stan2Py(STANfile, data, name="Polypharmacy")

    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMC = np.array(EX.MCMCdf)[:, 1:]

    EX.G = 9
    EX.L = 500

    A = scipy.linalg.block_diag(*([np.array([1])] * EX.L))
    B = np.zeros((EX.L, EX.G))
    C = np.ones((EX.G, EX.G + EX.L))
    EX.Tmask = np.tril(np.block([[A, B], [C]]))
    EX.Tmask = torch.tensor(EX.Tmask)

    return EX


def init_Polypharmacy_t():
    root = "".join([os.getcwd(), "/Examples/Linear Mixed Effects/Polypharmacy/"])
    STANfile = "".join([root, "model_polypharm_t.stan"])
    datfile = "".join([root, "Polypharmacy_data.rds"])
    MCMCfile = "".join([root, "Poly_MCMC_full_t.csv"])
    data = rds2dict(datfile)
    EX = Stan2Py(STANfile, data, name="Polypharmacy_t")

    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMC = np.array(EX.MCMCdf)[:, 1:]

    EX.G = 9
    EX.L = 500

    A = scipy.linalg.block_diag(*([np.array([1])] * EX.L))
    B = np.zeros((EX.L, EX.G))
    C = np.ones((EX.G, EX.G + EX.L))
    EX.Tmask = np.tril(np.block([[A, B], [C]]))
    EX.Tmask = torch.tensor(EX.Tmask)

    return EX


def init_GBP():
    root = "".join([os.getcwd(), "/Examples/State Space Models/GBP/"])
    STANfile = "".join([root, 'model_svm.stan'])
    datfile = "".join([root, 'gbpdata.rds'])
    data = rds2dict(datfile)
    EX = Stan2Py(STANfile, data, name="sv_GBP")
    p = 948
    msk = (np.triu(np.ones((p, p)), k=0) - np.triu(np.ones((p, p)), k=2)).T

    msk[-3:, :] = 1
    msk[-2, -1] = 0
    msk[-3, -1] = 0
    msk[-3, -2] = 0
    MCMCfile = "".join([root, 'GBP_MCMC_FULL.csv'])

    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMCnames = list(EX.MCMCdf.columns)
    EX.MCMC = EX.MCMCdf.to_numpy()
    EX.MCMC = EX.MCMC[:,1:]
    # EX.Tmask = msk
    EX.Tmask = torch.tensor(msk)
    EX.MCMCind = [945, 946, 947]

    return EX


def init_NYSE():
    root = "".join([os.getcwd(), "/Examples/State Space Models/NYSE/"])
    STANfile = "".join([root, 'model_svm.stan'])
    datfile = "".join([root, 'data_nyse.rds'])
    data = rds2dict(datfile)
    EX = Stan2Py(STANfile, data, name="sv_nyse")
    p = 2003
    msk = (np.triu(np.ones((p, p)), k=0) - np.triu(np.ones((p, p)), k=2)).T

    msk[-3:, :] = 1
    msk[-2, -1] = 0
    msk[-3, -1] = 0
    msk[-3, -2] = 0
    MCMCfile = "".join([root, 'NYSE_MCMC_FULL.csv'])

    EX.MCMCdf = pd.read_csv(MCMCfile)
    EX.MCMCnames = list(EX.MCMCdf.columns)
    EX.MCMC = EX.MCMCdf.to_numpy()
    EX.MCMC = EX.MCMC[:,1:]
    # EX.Tmask = msk
    EX.Tmask = torch.tensor(msk)
    EX.MCMCind = [2000, 2001, 2002]

    return EX