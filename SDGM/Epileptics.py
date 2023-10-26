"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains codes for example "Epilepsy" with normal random effects:
1) GVA
2) SDGM
3) SDGM_Centered
4) GVA+SAS(sinh-acrsinh transformation)
5) SDGM+SAS

Rob Salomone, Yu Xuejun, Sept 2022.
"""
from examples import init_Epilepsy_Slope
import numpy as np
import scipy
import torch as t
import time
import matplotlib.pyplot as plt
from Utils.PLOTtools import plot_marginals,plot_randeff,plot_ELBO
# --- this bit allows you to use files located in the Utils directory ----
import sys
sys.path.append("Utils/")
# -----------------------------------------------------------------------

sd = 1234567 # random seed

EX = init_Epilepsy_Slope()

logp = lambda th: t.tensor(EX.logp(th.detach().numpy()))
gr_logp = lambda th: t.tensor(EX.gr_logp(th.detach().numpy())).flatten()


start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from Gaussian_AD import SG_AD
print("---------------------Starts VB_GVA-------------------------")
VB_GVA = SG_AD(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask, up_a = False,optimizer="Adam")


VB_GVA.lr = 0.01
VB_GVA.full_train(num_steps=20000, show = 100)
VB_GVA.lr = 0.001
VB_GVA.full_train(num_steps=20000, show = 100)
VB_GVA.lr = 0.0001
VB_GVA.full_train(num_steps=10000, show = 100)

VB_GVA_time = time.time() - start_time
smp_GVA = VB_GVA.sample_set(n=10000)
smp_GVA= EX.constrain_samples(smp_GVA)



start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SG_AD import SG_AD
print("---------------------Starts VB_SDGM0-------------------------")
VB_SDGM0 = SG_AD(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask, up_a = True,optimizer="Adam")

VB_SDGM0.lr = 0.01
VB_SDGM0.full_train(num_steps=20000, show = 100)
VB_SDGM0.lr = 0.001
VB_SDGM0.full_train(num_steps=20000, show = 100)
VB_SDGM0.lr = 0.0001
VB_SDGM0.full_train(num_steps=10000, show = 100)

VB_SDGM0_time = time.time() - start_time
smp_SDGM0 = VB_SDGM0.sample_set(n=10000)
smp_SDGM0= EX.constrain_samples(smp_SDGM0)




start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from GVA_PLUS_SAS import SGGM
print("---------------------Starts VB_GC_SAS-------------------------")
VB_GVA_SAS = SGGM(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask, up_a=False,up_kurtosis=True,optimizer="Adam")


VB_GVA_SAS.lr = 0.01
VB_GVA_SAS.full_train(num_steps=20000, show = 100)
VB_GVA_SAS.lr = 0.001
VB_GVA_SAS.full_train(num_steps=20000, show = 100)
VB_GVA_SAS.lr = 0.0001
VB_GVA_SAS.full_train(num_steps=10000, show = 100)

VB_GVA_SAS_time = time.time() - start_time
smp_GVA_SAS = VB_GVA_SAS.sample_set(n=10000)
smp_GVA_SAS= EX.constrain_samples(smp_GVA_SAS)



start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SGGM_PLUS_SAS import SGGM
print("---------------------Starts VB_SDGM_SAS-------------------------")
VB_SDGM_SAS = SGGM(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask, up_a=True,up_kurtosis=True,optimizer="Adam")

VB_SDGM_SAS.lr = 0.01
VB_SDGM_SAS.full_train(num_steps=20000, show = 100)
VB_SDGM_SAS.lr = 0.001
VB_SDGM_SAS.full_train(num_steps=20000, show = 100)
VB_SDGM_SAS.lr = 0.0001
VB_SDGM_SAS.full_train(num_steps=10000, show = 100)

VB_SDGM_SAS_time = time.time() - start_time
smp_SDGM_SAS = VB_SDGM_SAS.sample_set(n=10000)
smp_SDGM_SAS= EX.constrain_samples(smp_SDGM_SAS)


t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SGC_AD import SGC_AD
start_time = time.time()
print("---------------------Starts VB_SDGM-------------------------")
VB_SDGM = SGC_AD(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask,optimizer="Adam")

VB_SDGM.prm['mu_'] = VB_SDGM_SAS.prm['mu']
VB_SDGM.prm['mu_'] = VB_SDGM.prm['mu_'].detach()
VB_SDGM.prm['mu_'].requires_grad_(True)
VB_SDGM.prm['a_'] = VB_SDGM_SAS.prm['a']
VB_SDGM.prm['a_'] = VB_SDGM.prm['a_'].detach()
VB_SDGM.prm['a_'].requires_grad_(True)

VB_SDGM.lr = 0.05
VB_SDGM.full_train(num_steps=10000, show = 100)
VB_SDGM.lr = 0.01
VB_SDGM.full_train(num_steps=10000, show = 100)
VB_SDGM.lr = 0.001
VB_SDGM.full_train(num_steps=20000, show = 100)
VB_SDGM.lr = 0.0001
VB_SDGM.full_train(num_steps=10000, show = 100)

VB_SDGM_time = time.time() - start_time
smp_SDGM = VB_SDGM.sample_set(n=10000)
smp_SDGM= EX.constrain_samples(smp_SDGM)

plt.figure(1)
ids = [118,119,120,121,122,123,124,125,126]
plt.clf()
name_list = [r"$\beta_0$",r"$\beta_{Base}}$",r"$\beta_{Trt}$",r"$\beta_{Base x Trt}$",r"$\beta_{Age}$",r"$\beta_{Visit}$",r"$\zeta_1$",r"$\zeta_2$",r"$\zeta_3$"]
plot_marginals(smp_GVA[:,ids],smp_SDGM0[:,ids],smp_SDGM[:,ids],smp_SDGM_SAS[:,ids],smp_GVA_SAS[:,ids],EX.MCMC[:,ids],
               num=9, labels = name_list)
# plt.legend(['SDGM','SDGM_C',"SDGM+SAS","GC",'MCMC'], loc = 'best')#,  bbox_to_anchor=(2.2, 0))
plt.savefig(f"Epilepsy_m.pdf")

#

plt.figure(2)
plt.clf()
ids_rand = list(range(59))
plot_randeff(EX.MCMC[:,ids_rand], smp_GVA[:,ids_rand],smp_SDGM0[:,ids_rand],smp_SDGM[:,ids_rand],smp_SDGM_SAS[:,ids_rand],smp_GVA_SAS[:,ids_rand],
             mean=True,labels=['GVA','SDGM','SDGM_C','SDGM+SAS','GVA+SAS'])
plt.savefig(f"Epilepsy_r1.pdf")

plt.figure(4)
plt.clf()
ids_rand = list(range(59,118))
plot_randeff(EX.MCMC[:,ids_rand], smp_GVA[:,ids_rand],smp_SDGM0[:,ids_rand],smp_SDGM[:,ids_rand],smp_SDGM_SAS[:,ids_rand],smp_GVA_SAS[:,ids_rand],
             mean=True,labels=['GVA','SDGM','SDGM_C','SDGM+SAS','GVA+SAS'])
plt.savefig(f"Epilepsy_r2.pdf")
#

plt.figure(3)
plot_ELBO(VB_GVA.average_ELBOlog,VB_SDGM0.average_ELBOlog,VB_SDGM.average_ELBOlog,VB_SDGM_SAS.average_ELBOlog,VB_GVA_SAS.average_ELBOlog,
          ylim=[2500, 3300],subx=[400, 500],suby=[3253.5, 3256],
          labels=['GVA','SDGM ','SDGM_C','SDGM+SAS','GVA+SAS'])
plt.savefig(f"Epilepsy_ELBO.pdf")

print("------the end-------")
print("VB_GVA_time =",VB_GVA_time)
print("VB_SDGM0_time =",VB_SDGM0_time)
print("VB_SDGM_time =",VB_SDGM_time)
print("VB_SDGMSAS_time =", VB_SDGM_SAS_time)
print("VB_GVASAS_time =", VB_GVA_SAS_time)

"""
VB_GVA_time = 42.34538912773132
VB_SDGM0_time = 57.5086030960083
VB_SDGM_time = 63.0853066444397
VB_SDGMSAS_time = 96.19118237495422
VB_GVASAS_time = 74.2280044555664
"""