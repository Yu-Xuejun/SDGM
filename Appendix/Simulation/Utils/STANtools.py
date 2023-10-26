"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains: Associated Tools to Interface with Stan Models.

Rob Salomone, Yu Xuejun, Sept 2022.
"""

import io

import numpy as np
import pystan
import pickle
from hashlib import md5
import torch
import os

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

class Stan2Py:
    def __init__(self, filename, data, name = []):
        # create a dictionary of parameters and their sizes
        model_code = open(filename, 'r').read()
        self.data = data
        self.stanmodel = StanModel_cache(model_code, model_name = name)
        self.stanfit = self.stanmodel.sampling(data=data, iter=10,
                                      chains=1, verbose=False,
                                      control={'adapt_delta' : 0.9})
        
        self.constrained_names = self.stanfit.flatnames[
                            :len(self.stanfit.constrained_param_names())]
        self.unconst_names = self.stanfit.flatnames[
                            :len(self.stanfit.unconstrained_param_names())]
        self.num_unconstrained = len(self.stanfit.unconstrained_param_names())
        self.num_constrained = len(self.stanfit.constrained_param_names())


    def logp(self, param_vec, updated_data=None):
        if updated_data is None:
            stanfit = self.stanfit
        else:
            # with suppress_stdout_stderr():
                stanfit = self.stanmodel.sampling(data=updated_data, iter=2,
                                                  chains=1, verbose=False,
                                                  control={'adapt_delta': 0.9})

        try:
            return stanfit.log_prob(param_vec)
        except:
            print('STAN logp error:', param_vec)

    def gr_logp(self, param_vec, updated_data=None):
        if updated_data is None:
            stanfit = self.stanfit
        else:
            # with suppress_stdout_stderr():
                stanfit = self.stanmodel.sampling(data=updated_data, iter=2,
                                                chains=1, verbose=False,
                                                control={'adapt_delta': 0.9})

        try:
            return stanfit.grad_log_prob(param_vec)
        except:
            print('STAN gr_logp error:', param_vec)
            return np.zeros_like(param_vec)
        

    def constrain_samples(self, samples):
        ''' Takes in Unconstrained Matrix, returns constrained Matrix'''
        constrained_size = len(self.stanfit.constrained_param_names())
        samples_new = np.zeros((samples.shape[0], constrained_size))
        
        for i in range(samples.shape[0]):
            samples_new[i,:] = self.stanfit.constrain_pars(samples[i,:])[:constrained_size]

        return samples_new
    
def StanModel_cache(model_code, model_name=None, **kwargs):
    ''' 
    Checks to see if there is a compiled version of the stan model
    before compiling to save time. 
    '''
    
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    #code_hash = md5(model_code.encode(encoding='UTF-8')).hexdigest()
    if model_name is None:
        cache_fn = 'STAN-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'STAN-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using saved Stan model.")

    # sm = pystan.StanModel(model_code=model_code)
    # with open(cache_fn, 'wb') as f:
    #     pickle.dump(sm, f)

    return sm


def Dict2Array(stanfit):
    dct = stanfit.extract(permuted = True)
    samples = np.column_stack([dct[item][:,0].reshape(-1,1) for item in dct])
    return samples

def cVec(dct): 
    for key in dct.keys(): 
        if dct[key].shape == ():  
            dct[key] = np.array(dct[key]).reshape(-1)
        
    return dct
