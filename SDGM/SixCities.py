"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains codes for example "Six Cities" with normal random effects:
1) GVA
2) SDGM
3) SDGM_Centered
4) GVA+SAS(sinh-acrsinh transformation)
5) SDGM+SAS

Rob Salomone, Yu Xuejun, Sept 2022.
"""
from examples import init_SixCities
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

EX = init_SixCities()

logp = lambda th: t.tensor(EX.logp(th.detach().numpy()))
gr_logp = lambda th: t.tensor(EX.gr_logp(th.detach().numpy())).flatten()

start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from Gaussian_AD import SG_AD
print("---------------------Starts GVA_AD-------------------------")
VB_GVA = SG_AD(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask, up_a=False,optimizer="Adam")

VB_GVA.lr = 0.05
VB_GVA.full_train(num_steps=20000, show = 100)
VB_GVA.lr = 0.01
VB_GVA.full_train(num_steps=20000, show = 100)
VB_GVA.lr = 0.001
VB_GVA.full_train(num_steps=10000, show = 100)
VB_GVA_time = time.time() - start_time
smp_GVA = VB_GVA.sample_set(n=10000)
smp_GVA= EX.constrain_samples(smp_GVA)



start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SG_AD import SG_AD
print("---------------------Starts SDGM0_AD-------------------------")
VB_SDGM0 = SG_AD(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask,optimizer="Adam")

VB_SDGM0.lr = 0.05
VB_SDGM0.full_train(num_steps=20000, show = 100)
VB_SDGM0.lr = 0.01
VB_SDGM0.full_train(num_steps=20000, show = 100)
VB_SDGM0.lr = 0.001
VB_SDGM0.full_train(num_steps=10000, show = 100)
VB_SDGM0_time = time.time() - start_time
smp_SDGM0 = VB_SDGM0.sample_set(n=10000)
smp_SDGM0= EX.constrain_samples(smp_SDGM0)


start_time = time.time()
t.manual_seed(sd) # set PyTorch's random seed
np.random.seed(sd)
from SGC_AD import SGC_AD
print("---------------------Starts SDGM_C_AD-------------------------")
VB_SDGM = SGC_AD(p = EX.num_unconstrained,
                logp =  logp, gr_logp = gr_logp,
                 mask = EX.Tmask,optimizer="Adam")

VB_SDGM.lr = 0.05
VB_SDGM.full_train(num_steps=20000, show = 100)
VB_SDGM.lr = 0.01
VB_SDGM.full_train(num_steps=20000, show = 100)
VB_SDGM.lr = 0.001
VB_SDGM.full_train(num_steps=10000, show = 100)

VB_SDGM_time = time.time() - start_time
smp_SDGM = VB_SDGM.sample_set(n=10000)
smp_SDGM= EX.constrain_samples(smp_SDGM)


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
VB_GVA_SAS.lr = 0.005
VB_GVA_SAS.full_train(num_steps=20000, show = 100)
VB_GVA_SAS.lr = 0.01
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

VB_SDGM_SAS.lr = 0.05
VB_SDGM_SAS.full_train(num_steps=20000, show = 100)
VB_SDGM_SAS.lr = 0.01
VB_SDGM_SAS.full_train(num_steps=20000, show = 100)
VB_SDGM_SAS.lr = 0.001
VB_SDGM_SAS.full_train(num_steps=10000, show = 100)
VB_SDGM_SAS_time = time.time() - start_time
smp_SDGM_SAS = VB_SDGM_SAS.sample_set(n=10000)
smp_SDGM_SAS= EX.constrain_samples(smp_SDGM_SAS)



plt.figure(1)
ids = [537,538,539,540,541]
plt.clf()
name_list = [r"$\beta_0$",r"$\beta_{Smoke}$",r"$\beta_{Age}$",r"$\beta_{Smoke x Age}$",r"$\zeta$"]
plot_marginals(smp_GVA[:,ids],smp_SDGM0[:,ids],smp_SDGM[:,ids],smp_SDGM_SAS[:,ids],smp_GVA_SAS[:,ids],EX.MCMC[:,ids],
               num=5, labels = name_list,plot_shape=[2,3],fig_size=[6,4])
plt.legend(['GVA','SDGM','SDGM_C',"SDGM+SAS","GVA+SAS",'MCMC'], loc = 'lower right',  bbox_to_anchor=(2.2, 0))
plt.savefig(f"SixCities_m.pdf")
#
plt.figure(2)
plt.clf()
ids_rand = list(range(537))
plot_randeff(EX.MCMC[:,ids_rand], smp_GVA[:,ids_rand],smp_SDGM0[:,ids_rand],smp_SDGM[:,ids_rand],smp_SDGM_SAS[:,ids_rand],smp_GVA_SAS[:,ids_rand],
             mean=True,labels=['GVA','SDGM','SDGM_C','SDGM_SAS','GVA_SAS'])
plt.savefig(f"SixCities_r.pdf")
#
plt.figure(3)
plot_ELBO(VB_GVA.average_ELBOlog,VB_SDGM0.average_ELBOlog,VB_SDGM.average_ELBOlog,VB_SDGM_SAS.average_ELBOlog,VB_GVA_SAS.average_ELBOlog,
          ylim=[-1000, -250],subx=[410, 480],suby=[-325, -305],
          labels=['GVA','SDGM ','SDGM_C','SDGM+SAS','GVA+SAS'])
plt.savefig(f"SixCities_ELBO.pdf")

print("------the end-------")
print("VB_GVA_time =",VB_GVA_time)
print("VB_SDGM0_time =",VB_SDGM0_time)
print("VB_SDGM_time =",VB_SDGM_time)
print("VB_SDGMSAS_time =", VB_SDGM_SAS_time)
print("VB_GVASAS_time =", VB_GVA_SAS_time)
"""
VB_GVA_time = 80.73035264015198
VB_SDGM0_time = 99.04013872146606
VB_SDGM_time = 103.3491530418396
VB_SDGMSAS_time = 145.3409080505371
VB_GVASAS_time = 117.47318410873413
"""

