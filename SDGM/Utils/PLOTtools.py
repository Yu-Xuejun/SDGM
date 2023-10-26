"""
Codes for paper
"Structured variational approximations with skew normal decomposable graphical models".

This file contains codes for plots:
1) kernel density plot of marginal distributions
2) mean, std and skewness scatter plots (VB vs MCMC) for random effects
3) ELBO plots
4) GVA+SAS(sinh-acrsinh transformation)
5) SDGM+SAS

Rob Salomone, Yu Xuejun, Sept 2022.
"""
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import numpy as np
import scipy.stats as stats
import scipy
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_marginals(data, data2 =[], data3 = [], data4=[], data5=[],data6=[], ids =[] ,num=[], bw = 'silverman', labels=[], plot_shape=[],fig_size=[]):
    plt.clf()

    if any(ids): ids = list(ids)

    assert((num or ids)), "num or ids must be specified"
    if not num:
        num = len(ids)
    if not ids:
        ids = list(range(num))

    if fig_size != []:
        fig = plt.figure(figsize=(fig_size[0],fig_size[1]))
    else:
        fig = plt.figure()
    for i in range(1,num+1):
        if plot_shape == []:
            ax = fig.add_subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i)
        else:
            ax = fig.add_subplot(plot_shape[0], plot_shape[1], i)
        if len(labels) > 0: ax.set_title(labels[ids[i-1]])
        # Compute the kernel density estimate
        x, y = FFTKDE(kernel='gaussian', bw=bw).fit(data[:,ids[i-1]]).evaluate()
        plt.plot(x, y, '#2CBDFE', label='KDE /w ISJ')
        
        if data2 != []:
            x, y = FFTKDE(kernel='gaussian', bw=bw).fit(data2[:,ids[i-1]]).evaluate()
            plt.plot(x, y, 'tomato', label='KDE /w ISJ')
        if data3 != []:
            x, y = FFTKDE(kernel='gaussian', bw=bw).fit(data3[:,ids[i-1]]).evaluate()
            plt.plot(x, y, '#F5B14C', label='KDE /w ISJ')
        if data4 != []:
            x, y = FFTKDE(kernel='gaussian', bw=bw).fit(data4[:,ids[i-1]]).evaluate()
            plt.plot(x, y, '#9D2EC5', label='KDE /w ISJ')
            
        if data5 != []:
            x, y = FFTKDE(kernel='gaussian', bw=bw).fit(data5[:,ids[i-1]]).evaluate()
            plt.plot(x, y, 'yellowgreen', label='KDE /w ISJ')

        if data6 != []:
            x, y = FFTKDE(kernel='gaussian', bw=bw).fit(data6[:,ids[i-1]]).evaluate()
            plt.plot(x, y, 'black', label='KDE /w ISJ')
         
        plt.tight_layout()
               

def plot_randeff(data_mcmc, data_vb, data_vb2 = [], data_vb3=[], data_vb4=[], data_vb5=[], mean=True, sd=False, ids =[] ,num=[], bw = 'silverman', labels=[]):
    # plot posterior mean, std and skewness of random parameters

    plt.clf()
    fig = plt.figure()
    if mean:
        ax = fig.add_subplot(len(labels), 3, 1)
        x_mean, y_mean = np.mean(data_mcmc,0), np.mean(data_vb,0)
        ax.plot(x_mean, x_mean, 'o',alpha=0.2)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r-',  zorder=0)
        ax.set_title('mean')
        if len(labels) > 0: ax.set_ylabel(labels[0])
        if len(labels) == 1: ax.set_xlabel("MCMC")

        ax = fig.add_subplot(len(labels), 3, 2)
        x_std, y_std = np.std(data_mcmc, 0), np.std(data_vb, 0)
        ax.plot(x_std, y_std, 'o', alpha=0.2)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
        ax.set_title('std')
        if len(labels) == 1: ax.set_xlabel("MCMC")

        ax = fig.add_subplot(len(labels), 3, 3)
        x_skewness, y_skewness = scipy.stats.skew(data_mcmc, 0), scipy.stats.skew(data_vb, 0)
        ax.plot(x_skewness, y_skewness, 'o' ,alpha=0.2)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
        ax.set_title('skewness')
        if len(labels) == 1: ax.set_xlabel("MCMC")
        if data_vb2 != []:
            ax = fig.add_subplot(len(labels), 3, 4)
            ax.plot(np.mean(data_mcmc, 0), np.mean(data_vb2, 0), 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
            if len(labels) > 1: ax.set_ylabel(labels[1])
            if len(labels) == 2: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 5)
            x_std, y_std = np.std(data_mcmc, 0), np.std(data_vb2, 0)
            ax.plot(x_std, y_std, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-',c='tomato', zorder=0)
            if len(labels) == 2: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 6)
            x_skewness, y_skewness = scipy.stats.skew(data_mcmc, 0), scipy.stats.skew(data_vb2, 0)
            ax.plot(x_skewness, y_skewness, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
            if len(labels) == 2: ax.set_xlabel("MCMC")
        if data_vb3 != []:
            ax = fig.add_subplot(len(labels), 3, 7)
            ax.plot(np.mean(data_mcmc, 0), np.mean(data_vb3, 0), 'o',alpha=0.3)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
            if len(labels) > 2: ax.set_ylabel(labels[2])
            if len(labels) == 3: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 8)
            x_std, y_std = np.std(data_mcmc, 0), np.std(data_vb3, 0)
            ax.plot(x_std, y_std, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato',zorder=0)
            if len(labels) == 3: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 9)
            x_skewness, y_skewness = scipy.stats.skew(data_mcmc, 0), scipy.stats.skew(data_vb3, 0)
            ax.plot(x_skewness, y_skewness, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
            if len(labels) == 3: ax.set_xlabel("MCMC")
        if data_vb4 != []:
            ax = fig.add_subplot(len(labels), 3, 10)
            ax.plot(np.mean(data_mcmc, 0), np.mean(data_vb4, 0), 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
            if len(labels) > 3: ax.set_ylabel(labels[3])
            if len(labels) == 4: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 11)
            x_std, y_std = np.std(data_mcmc, 0), np.std(data_vb4, 0)
            ax.plot(x_std, y_std, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato',zorder=0)
            if len(labels) == 4: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 12)
            x_skewness, y_skewness = scipy.stats.skew(data_mcmc, 0), scipy.stats.skew(data_vb4, 0)
            ax.plot(x_skewness, y_skewness, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-',c='tomato', zorder=0)
            if len(labels) == 4: ax.set_xlabel("MCMC")
        if data_vb5 != []:
            ax = fig.add_subplot(len(labels), 3, 13)
            ax.plot(np.mean(data_mcmc, 0), np.mean(data_vb5, 0), 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato', zorder=0)
            if len(labels) > 4: ax.set_ylabel(labels[4])
            if len(labels) == 5: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 14)
            x_std, y_std = np.std(data_mcmc, 0), np.std(data_vb5, 0)
            ax.plot(x_std, y_std, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-', c='tomato',zorder=0)
            if len(labels) == 5: ax.set_xlabel("MCMC")
            ax = fig.add_subplot(len(labels), 3, 15)
            x_skewness, y_skewness = scipy.stats.skew(data_mcmc, 0), scipy.stats.skew(data_vb5, 0)
            ax.plot(x_skewness, y_skewness, 'o',alpha=0.2)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, 'r-',c='tomato', zorder=0)
            if len(labels) == 5: ax.set_xlabel("MCMC")

        plt.tight_layout()


def plot_ELBO(ELBO1,ELBO2=[],ELBO3=[],ELBO4=[],ELBO5=[],ylim=[],subx=[],suby=[],labels=[]):
    plt.clf()
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(ELBO1, '#2CBDFE')
    if ELBO2 != []:
        ax.plot(ELBO2, 'tomato')
    if ELBO3 != []:
        ax.plot(ELBO3, '#F5B14C')
    if ELBO4 != []:
        ax.plot(ELBO4, '#9D2EC5',alpha=0.7)
    if ELBO5 != []:
        ax.plot(ELBO5, 'yellowgreen',alpha=0.8)
    if ylim != []:
        ax.set_ylim(ylim[0], ylim[1])

    ax.legend(labels=labels, ncol=4)

    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.5, 0.3, 1, 1),
                       bbox_transform=ax.transAxes)

    axins.plot(ELBO1, '#2CBDFE')
    if ELBO2 != []:
        axins.plot(ELBO2, 'tomato')
    if ELBO3 != []:
        axins.plot(ELBO3, '#F5B14C')
    if ELBO4 != []:
        axins.plot(ELBO4, '#9D2EC5',alpha=0.7)
    if ELBO5 != []:
        axins.plot(ELBO5, 'yellowgreen',alpha=0.8)
    # 调整子坐标系的显示范围
    if subx != []:
        axins.set_xlim(subx[0], subx[1])
    if suby != []:
        axins.set_ylim(suby[0], suby[1])
    # loc1 loc2: 坐标系的四个角
    # 1 (右上) 2 (左上) 3(左下) 4(右下)
    mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec='k', lw=1,zorder=3)
    ax.set_xlabel("iterations(x 100)")
    ax.set_title("Evidence Lower Bound")


