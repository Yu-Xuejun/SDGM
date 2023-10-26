# Structured variational approximations with skew normal decomposable graphical models

This repository contains code resources to accompany our research paper,

>Structured variational approximations with skew normal decomposable graphical models.

These codes carried out experiments of noncentered GVA+SAS method on Epilepsy data in Author Response.

Detailed steps to run the code are as follows.

- 1 

  + Step 1.1: Run [Examples\Linear Mixed Effects\Epilepsy\Epilepsy.R] and  [Examples\Linear Mixed Effects\Epilepsy\Epilepsy_t.R] to get MCMC results which will be used in the following python code.
  + Step 1.2: Run [Epileptics.py] to get GVA, SDGM, SDGM-C, GVA-SAS-L1, GVA-SAS-L2, SDGM-SAS results for normal distributed random effects. The generated file "Epilepsy_r1_SAS.pdf" gives figure 1 in Author response.
