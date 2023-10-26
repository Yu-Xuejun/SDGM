# Structured variational approximations with skew normal decomposable graphical models

This repository contains code resources to accompany our research paper,

>Structured variational approximations with skew normal decomposable graphical models.

These codes carried out simulation experiments of Six Cities data in Appendix E.

Detailed steps to run the code are as follows.

- 1 Generate simulation and MCMC results

  + Step 1.1: Run [Examples\Linear Mixed Effects\SixCities\synthetic_dataset.R] to get simulated synthetic data.
  + Step 1.2: Run [Examples\Linear Mixed Effects\SixCities\Rstan_simulation.R] to get MCMC results.
  + Step 1.3: Run [Simulation-SixCities.py] four times with n=50, n=100, n=250, n=750 (need to be adjusted by hand) to get SDGM, SDGM-C, GVA-SAS, SDGM-SAS results saved.
  + Step 1.4: Run [Boxplot-simulation.py] to get figure 13 in the Appendix E.
