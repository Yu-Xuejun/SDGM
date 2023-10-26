"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains codes for example "Six Cities" with simulated data.
1) SDGM
2) SDGM_Centered
3) GVA+SAS(sinh-acrsinh transformation)
4) SDGM+SAS

Yu Xuejun, July 2023.
"""
from examples import init_SixCities_simulation,import_mcmc_result,import_test_data
from Simulation_Study import prediction_evaluation
import numpy as np
import scipy
import torch as t
import time
# --- this bit allows you to use files located in the Utils directory ----
import sys
sys.path.append("Utils/")
# -----------------------------------------------------------------------

sd = 1234567 # random seed

# Setting n to get results of certain size of simulated data.
# n = 50
# n = 100
# n = 250
n = 750
R = 10
len_ELBO = 150
ids_rand = list(range(n))
ids_fix = list(range(n,n+4))
EX_list = []
TD_list = []
for r in range(1, R + 1):
    print(r)
    obj = init_SixCities_simulation(n, r)
    EX_list.append(obj)
    td = import_test_data(n,r)
    TD_list.append(td)

MCMC_moments = import_mcmc_result(n,R)

def moments_comparison(MCMC_moments,VB_moments, n, R):
    # A = np.abs(MCMC_moments["mu_mcmc"] - VB_moments["mu"]).sum()/(n*R)
    # B = np.abs(MCMC_moments["sigma_mcmc"]- VB_moments["sigma"]).sum()/(n*R)
    # C = np.abs(MCMC_moments["k_mcmc"] - VB_moments["k"]).sum()/(n*R)
    Mean_vec = np.zeros(3)
    SE_vec = np.zeros(3)
    A = np.abs(MCMC_moments["mu_mcmc"] - VB_moments["mu"]).sum(axis=1)/n
    # Mean_vec[0] = np.mean(A)
    # SE_vec[0] = np.std(A)
    B = np.abs(MCMC_moments["sigma_mcmc"]- VB_moments["sigma"]).sum(axis=1)/n
    # Mean_vec[1] = np.mean(B)
    # SE_vec[1] = np.std(B)
    C = np.abs(MCMC_moments["k_mcmc"] - VB_moments["k"]).sum(axis=1)/n
    # Mean_vec[2] = np.mean(C)
    # SE_vec[2] = np.std(C)
    return A, B, C

start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SG_AD import SG_AD
print("---------------------Starts SDGM0_AD-------------------------")
VB_SDGM0_mean = np.zeros( (R, n) )
VB_SDGM0_sd = np.zeros( (R, n) )
VB_SDGM0_skewness = np.zeros( (R, n) )
SDGM0_ELBOlog = np.zeros( (R, len_ELBO) )
VB_SDGM0_pll = np.zeros(R)
VB_SDGM0_acc = np.zeros(R)
VB_SDGM0_acc2 = np.zeros(R)

for i in range(R):
    logp = lambda th: t.tensor(EX_list[i].logp(th.detach().numpy()))
    gr_logp = lambda th: t.tensor(EX_list[i].gr_logp(th.detach().numpy())).flatten()

    VB_SDGM0 = SG_AD(p = n+5,
                    logp = logp, gr_logp = gr_logp,
                    mask = EX_list[i].Tmask,optimizer="Adam")

    VB_SDGM0.lr = 0.05
    VB_SDGM0.full_train(num_steps=5000, show = 100)
    VB_SDGM0.lr = 0.01
    VB_SDGM0.full_train(num_steps=5000, show = 100)
    VB_SDGM0.lr = 0.001
    VB_SDGM0.full_train(num_steps=5000, show = 100)
    smp_SDGM0 = VB_SDGM0.sample_set(n=5000)
    smp_SDGM0 = EX_list[i].constrain_samples(smp_SDGM0)

    VB_SDGM0_mean[i,:] = np.mean(smp_SDGM0[:,ids_rand], 0)
    VB_SDGM0_sd[i,:] = np.std(smp_SDGM0[:,ids_rand], 0)
    VB_SDGM0_skewness[i,:] = scipy.stats.skew(smp_SDGM0[:,ids_rand], 0)
    SDGM0_ELBOlog[i,:] = VB_SDGM0.average_ELBOlog

    test_dat = TD_list[i]
    VB_SDGM0_pll[i],VB_SDGM0_acc[i],VB_SDGM0_acc2[i] = prediction_evaluation(test_dat["y_test"], test_dat["X"], smp_SDGM0[:,ids_fix], smp_SDGM0[:,ids_rand])



VB_SDGM0_time = time.time() - start_time
VB_SDGM0_moments = {"mu":VB_SDGM0_mean, "sigma":VB_SDGM0_sd, "k":VB_SDGM0_skewness}
VB_SDGM0_mu, VB_SDGM0_sigma, VB_SDGM0_k = moments_comparison(MCMC_moments,VB_SDGM0_moments, n, R)


VB_SDGM_mean = np.zeros( (R, n) )
VB_SDGM_sd = np.zeros( (R, n) )
VB_SDGM_skewness = np.zeros( (R, n) )
SDGM_ELBOlog = np.zeros( (R, len_ELBO) )
VB_SDGM_pll = np.zeros(R)
VB_SDGM_acc = np.zeros(R)
VB_SDGM_acc2 = np.zeros(R)
start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SGC_AD import SGC_AD
print("---------------------Starts SDGM0_AD-------------------------")
for i in range(R):
    logp = lambda th: t.tensor(EX_list[i].logp(th.detach().numpy()))
    gr_logp = lambda th: t.tensor(EX_list[i].gr_logp(th.detach().numpy())).flatten()
    
    VB_SDGM = SGC_AD(p = n+5,
                    logp =  logp, gr_logp = gr_logp,
                    mask = EX_list[i].Tmask,optimizer="Adam")

    VB_SDGM.lr = 0.05
    VB_SDGM.full_train(num_steps=5000, show = 100)
    VB_SDGM.lr = 0.01
    VB_SDGM.full_train(num_steps=5000, show = 100)
    VB_SDGM.lr = 0.001
    VB_SDGM.full_train(num_steps=5000, show = 100)
    smp_SDGM = VB_SDGM.sample_set(n=5000)
    smp_SDGM = EX_list[i].constrain_samples(smp_SDGM)

    VB_SDGM_mean[i,:] = np.mean(smp_SDGM[:,ids_rand], 0)
    VB_SDGM_sd[i,:] = np.std(smp_SDGM[:,ids_rand], 0)
    VB_SDGM_skewness[i,:] = scipy.stats.skew(smp_SDGM[:,ids_rand], 0)
    SDGM_ELBOlog[i,:] = VB_SDGM.average_ELBOlog
    
    test_dat = TD_list[i]
    VB_SDGM_pll[i],VB_SDGM_acc[i],VB_SDGM_acc2[i] = prediction_evaluation(test_dat["y_test"], test_dat["X"], smp_SDGM[:,ids_fix], smp_SDGM[:,ids_rand])


VB_SDGM_time = time.time() - start_time
VB_SDGM_moments = {"mu":VB_SDGM_mean, "sigma":VB_SDGM_sd, "k":VB_SDGM_skewness}
VB_SDGM_mu, VB_SDGM_sigma, VB_SDGM_k  = moments_comparison(MCMC_moments,VB_SDGM_moments, n, R)


VB_GVA_SAS_mean = np.zeros( (R, n) )
VB_GVA_SAS_sd = np.zeros( (R, n) )
VB_GVA_SAS_skewness = np.zeros( (R, n) )
GVA_SAS_ELBOlog = np.zeros( (R, len_ELBO) )
VB_GVA_SAS_pll = np.zeros(R)
VB_GVA_SAS_acc = np.zeros(R)
VB_GVA_SAS_acc2 = np.zeros(R)
start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from GVA_PLUS_SAS import SGGM
print("---------------------Starts SDGM0_AD-------------------------")
for i in range(R):
    logp = lambda th: t.tensor(EX_list[i].logp(th.detach().numpy()))
    gr_logp = lambda th: t.tensor(EX_list[i].gr_logp(th.detach().numpy())).flatten()
    
    VB_GVA_SAS = SGGM(p = n+5,
                    logp =  logp, gr_logp = gr_logp,
                    mask = EX_list[i].Tmask,optimizer="Adam")

    VB_GVA_SAS.lr = 0.01
    VB_GVA_SAS.full_train(num_steps=5000, show = 100)
    VB_GVA_SAS.lr = 0.005
    VB_GVA_SAS.full_train(num_steps=5000, show = 100)
    VB_GVA_SAS.lr = 0.001
    VB_GVA_SAS.full_train(num_steps=5000, show = 100)
    smp_GVA_SAS = VB_GVA_SAS.sample_set(n=5000)
    smp_GVA_SAS= EX_list[i].constrain_samples(smp_GVA_SAS)

    VB_GVA_SAS_mean[i,:] = np.mean(smp_GVA_SAS[:,ids_rand], 0)
    VB_GVA_SAS_sd[i,:] = np.std(smp_GVA_SAS[:,ids_rand], 0)
    VB_GVA_SAS_skewness[i,:] = scipy.stats.skew(smp_GVA_SAS[:,ids_rand], 0)
    GVA_SAS_ELBOlog[i,:] = VB_GVA_SAS.average_ELBOlog
    
    test_dat = TD_list[i]
    VB_GVA_SAS_pll[i],VB_GVA_SAS_acc[i],VB_GVA_SAS_acc2[i] = prediction_evaluation(test_dat["y_test"], test_dat["X"], smp_GVA_SAS[:,ids_fix], smp_GVA_SAS[:,ids_rand])


VB_GVA_SAS_time = time.time() - start_time
VB_GVA_SAS_moments = {"mu":VB_GVA_SAS_mean, "sigma":VB_GVA_SAS_sd, "k":VB_GVA_SAS_skewness}
VB_GVA_SAS_mu, VB_GVA_SAS_sigma, VB_GVA_SAS_k = moments_comparison(MCMC_moments,VB_GVA_SAS_moments, n, R)



VB_SDGM_SAS_mean = np.zeros( (R, n) )
VB_SDGM_SAS_sd = np.zeros( (R, n) )
VB_SDGM_SAS_skewness = np.zeros( (R, n) )
SDGM_SAS_ELBOlog = np.zeros( (R, len_ELBO) )
VB_SDGM_SAS_pll = np.zeros(R)
VB_SDGM_SAS_acc = np.zeros(R)
VB_SDGM_SAS_acc2 = np.zeros(R)
start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SGGM_PLUS_SAS import SGGM
print("---------------------Starts SDGM0_AD-------------------------")
for i in range(R):
    logp = lambda th: t.tensor(EX_list[i].logp(th.detach().numpy()))
    gr_logp = lambda th: t.tensor(EX_list[i].gr_logp(th.detach().numpy())).flatten()
    
    VB_SDGM_SAS = SGGM(p = n+5,
                    logp =  logp, gr_logp = gr_logp,
                    mask = EX_list[i].Tmask,optimizer="Adam")

    VB_SDGM_SAS.lr = 0.05
    VB_SDGM_SAS.full_train(num_steps=5000, show = 100)
    VB_SDGM_SAS.lr = 0.01
    VB_SDGM_SAS.full_train(num_steps=5000, show = 100)
    VB_SDGM_SAS.lr = 0.001
    VB_SDGM_SAS.full_train(num_steps=5000, show = 100)
    smp_SDGM_SAS = VB_SDGM_SAS.sample_set(n=5000)
    smp_SDGM_SAS= EX_list[i].constrain_samples(smp_SDGM_SAS)

    VB_SDGM_SAS_mean[i,:] = np.mean(smp_SDGM_SAS[:,ids_rand], 0)
    VB_SDGM_SAS_sd[i,:] = np.std(smp_SDGM_SAS[:,ids_rand], 0)
    VB_SDGM_SAS_skewness[i,:] = scipy.stats.skew(smp_SDGM_SAS[:,ids_rand], 0)

    SDGM_SAS_ELBOlog[i,:] = VB_SDGM_SAS.average_ELBOlog

    test_dat = TD_list[i]
    VB_SDGM_SAS_pll[i],VB_SDGM_SAS_acc[i],VB_SDGM_SAS_acc2[i] = prediction_evaluation(test_dat["y_test"], test_dat["X"], smp_SDGM_SAS[:,ids_fix], smp_SDGM_SAS[:,ids_rand])



VB_SDGM_SAS_time = time.time() - start_time
VB_SDGM_SAS_moments = {"mu":VB_SDGM_SAS_mean, "sigma":VB_SDGM_SAS_sd, "k":VB_SDGM_SAS_skewness}
VB_SDGM_SAS_mu, VB_SDGM_SAS_sigma, VB_SDGM_SAS_k = moments_comparison(MCMC_moments,VB_SDGM_SAS_moments, n, R)


print("VB_SDGM0_time =",VB_SDGM0_time) # 170s 190s 316s 1021s
print("VB_SDGM_time =",VB_SDGM_time) # 182s 204s 328s 977s
print("VB_GVA_SAS_time =",VB_GVA_SAS_time) # 226s 247s 377s 1046s
print("VB_SDGM_SAS_time =",VB_SDGM_SAS_time) # 290s 325s 459s 1130s


mu_matrix = np.column_stack((VB_SDGM0_mu, VB_SDGM_mu, VB_GVA_SAS_mu, VB_SDGM_SAS_mu))
sigma_matrix = np.column_stack((VB_SDGM0_sigma, VB_SDGM_sigma, VB_GVA_SAS_sigma, VB_SDGM_SAS_sigma))
k_matrix = np.column_stack((VB_SDGM0_k, VB_SDGM_k, VB_GVA_SAS_k, VB_SDGM_SAS_k))
pll_matrix = np.column_stack((VB_SDGM0_pll, VB_SDGM_pll, VB_GVA_SAS_pll, VB_SDGM_SAS_pll))
acc_matrix = np.column_stack((VB_SDGM0_acc, VB_SDGM_acc, VB_GVA_SAS_acc, VB_SDGM_SAS_acc))

file_name = "Simulation-n{}.npz".format(n)
np.savez(file_name, mu_matrix=mu_matrix, sigma_matrix=sigma_matrix, k_matrix=k_matrix,
                    pll_matrix=pll_matrix, acc_matrix=acc_matrix)

# print("MCMC")
# print(MCMC_moments["pll"])
# print(MCMC_moments["acc"])
#
# print("SDGM0")
# print(VB_SDGM0_Mean_vec)
# print(VB_SDGM0_SE_vec)
# print(np.mean(VB_SDGM0_pll))
# print(np.mean(VB_SDGM0_acc))
# print("std")
# print(np.std(VB_SDGM0_pll))
# print(np.std(VB_SDGM0_acc))
#
# print("SDGM")
# print(VB_SDGM_Mean_vec)
# print(VB_SDGM_SE_vec)
# print(np.mean(VB_SDGM_pll))
# print(np.mean(VB_SDGM_acc))
# print("std")
# print(np.std(VB_SDGM_pll))
# print(np.std(VB_SDGM_acc))
#
# print("GVA+SAS")
# print(VB_GVA_SAS_Mean_vec)
# print(VB_GVA_SAS_SE_vec)
# print(np.mean(VB_GVA_SAS_pll))
# print(np.mean(VB_GVA_SAS_acc))
# print("std")
# print(np.std(VB_GVA_SAS_pll))
# print(np.std(VB_GVA_SAS_acc))
#
# print("SDGM+SAS")
# print(VB_SDGM_SAS_Mean_vec)
# print(VB_SDGM_SAS_SE_vec)
# print(np.mean(VB_SDGM_SAS_pll))
# print(np.mean(VB_SDGM_SAS_acc))
# print(np.std(VB_SDGM_SAS_pll))
# print(np.std(VB_SDGM_SAS_acc))