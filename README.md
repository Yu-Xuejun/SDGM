# Structured variational approximations with skew normal decomposable graphical models

This repository contains code resources to accompany our research paper,

>Structured variational approximations with skew normal decomposable graphical models.

In "SDGM" folder, the codes are carried out for experiments in Section 4: *Examples.*
In "Appendix" folder, the codes are for simulations in appendix. There is a "readme" file in the folder.

Detailed steps to run the code in "SDGM" folder are as follows.

- 1 Section 4.1: *Six cities data*

  + Step 1.1: Run [Examples\Linear Mixed Effects\SixCities\Rstan.R] to get MCMC results which will be used in the following python code.
  + Step 1.2: Run [SixCities.py] to get GVA, SDGM, SDGM-C, GVA-SAS, SDGM-SAS results for normal distributed random effects (Figure 1, Figure 5 in appendix D).
  + Step 1.3: Run [SixCities_t.py] to get GVA, SDGM, SDGM-C, GVA-SAS, SDGM-SAS results for t distributed random effects. (Figure 4 and 5 in appendix D)

- 2 Section 4.2: *Polypharmacy data*

   + Step 2.1: Run [Examples\Linear Mixed Effects\Polypharmacy\polypharmRplots.R] and  [Examples\Linear Mixed Effects\Polypharmacy\polypharm_t.R] to get MCMC results which will be used in the following python code.
   + Step 2.2: Run [Polypharmacy.py] to get GVA, SDGM, SDGM-C, GVA-SAS, SDGM-SAS results for normal distributed random effects. (Figure 6 and 8 in appendix D)
   + Step 2.3: Run [Polypharmacy_t.py] to get GVA, SDGM, SDGM-C, GVA-SAS, SDGM-SAS results for t distributed random effects. (Figure 7 and 8 in appendix D)

- 3 Section 4.3: *Epilepsy data*

   + Step 3.1: Run [Examples\Linear Mixed Effects\Epilepsy\Epilepsy.R] and  [Examples\Linear Mixed Effects\Epilepsy\Epilepsy_t.R] to get MCMC results which will be used in the following python code.
   + Step 3.2: Run [Epileptics.py] to get GVA, SDGM, SDGM-C, GVA-SAS, SDGM-SAS results for normal distributed random effects. (Figure 2, Figure 10 and 11 in appendix)
   + Step 3.3: Run [Epileptics_t.py] to get GVA, SDGM, SDGM-C, GVA-SAS, SDGM-SAS results for t distributed random effects. (Figure 9,10 and 11 in appendix D)

- 4 Section 4.4: *New York stock exchange data*

   + Step 4.1: Run [Examples\State Space Models\NYSE\NYSE.R] and  [Examples\State Space Models\NYSE\NYSE.R] to get MCMC results which will be used in the following python code.
   + Step 4.2: Run [NYSE.py] to get GVA, SDGM, SDGM-C, GVA-SAS, SDGM-SAS results. (Figure 3, Figure 12 in appendix D)
   + Step 4.3: Run [NYSE_sp.py] to get the same results with sparse matrix computation.