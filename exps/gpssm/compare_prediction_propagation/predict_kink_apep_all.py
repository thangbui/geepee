import sys
sys.path.append('../../../')
import numpy as np
import scipy.stats
import os, sys
import geepee.aep_models as aep
from geepee.config import PROP_MC, PROP_MM
import pdb
from scipy.misc import logsumexp
np.random.seed(0)

import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 488.13                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.8            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "text.fontsize": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

def predict_using_trained_models():    
    alphas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    M = 20
    K = 20
    T_test = 20
    mm_se = np.zeros((K, T_test, len(alphas)))
    mm_ll = np.zeros((K, T_test, len(alphas)))
    mc_se = np.zeros((K, T_test, len(alphas)))
    mc_ll = np.zeros((K, T_test, len(alphas)))
    for k in range(K):
        y_train = np.loadtxt('data/kink_train_%d.txt'%k)
        y_train = np.reshape(y_train, [y_train.shape[0], 1])
        y_test = np.loadtxt('data/kink_test_%d.txt'%k)
        y_test = np.reshape(y_test, [y_test.shape[0], 1])
        y_test = y_test[:, 0]
        
        for i, alpha in enumerate(alphas):
            # for prop_mode in [PROP_MM, PROP_MC]:
            for prop_mode in [PROP_MM]:
                print 'k %d, alpha = %.6f, prop_mode %s' % (k, alpha, prop_mode)
                model_aep = aep.SGPSSM(y_train, 1, M, 
                    lik='Gaussian', prior_mean=0, prior_var=1)
                model_aep.load_model('trained_models/kink_aep_model_index_%d_M_%d_alpha_%.4f_%s.pickle' % (k, M, alpha, prop_mode))
                
                # predict using MM and MC, TODO: lin
                _, _, my_MM, _, vy_MM = model_aep.predict_forward(T_test, prop_mode=PROP_MM)
                _, my_MC, vy_MC = model_aep.predict_forward(T_test, prop_mode=PROP_MC, no_samples=500)
                

                my_MM = my_MM[:, 0]
                vy_MM = vy_MM[:, 0]

                my_MC = my_MC[:, :, 0].T
                vy_MC = vy_MC[:, :, 0].T

                mm_se[k, :, i] = (my_MM - y_test)**2
                mm_ll[k, :, i] = -0.5 * np.log(2*np.pi*vy_MM) - 0.5*(my_MM-y_test)**2/vy_MM

                mc_se[k, :, i] = (np.mean(my_MC, axis=0) - y_test)**2
                mc_ll[k, :, i] = logsumexp(-0.5*np.log(2*np.pi*vy_MC) - 0.5*(my_MC-y_test)**2/vy_MC, axis=0) - np.log(my_MC.shape[0])             

    mm_se_mean = np.mean(mm_se, axis=0)
    mm_se_error = np.std(mm_se, axis=0) / np.sqrt(K)
    mm_ll_mean = np.mean(mm_ll, axis=0)
    mm_ll_error = np.std(mm_ll, axis=0) / np.sqrt(K)
    mc_se_mean = np.mean(mc_se, axis=0)
    mc_se_error = np.std(mc_se, axis=0) / np.sqrt(K)
    mc_ll_mean = np.mean(mc_ll, axis=0)
    mc_ll_error = np.std(mc_ll, axis=0) / np.sqrt(K)

    np.savetxt('res/kink_mm_se_mean.txt', mm_se_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mm_se_error.txt', mm_se_error, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mm_ll_mean.txt', mm_ll_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mm_ll_error.txt', mm_ll_error, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_se_mean.txt', mc_se_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_se_error.txt', mc_se_error, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_ll_mean.txt', mc_ll_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_ll_error.txt', mc_ll_error, fmt='%.5f', delimiter=',')           


def plot_res():
    mm_se_mean = np.loadtxt('res/kink_mm_se_mean.txt', delimiter=',')
    mm_se_error = np.loadtxt('res/kink_mm_se_error.txt', delimiter=',')
    mc_se_mean = np.loadtxt('res/kink_mc_se_mean.txt', delimiter=',')
    mc_se_error = np.loadtxt('res/kink_mc_se_error.txt', delimiter=',')
    mm_ll_mean = np.loadtxt('res/kink_mm_ll_mean.txt', delimiter=',')
    mm_ll_error = np.loadtxt('res/kink_mm_ll_error.txt', delimiter=',')
    mc_ll_mean = np.loadtxt('res/kink_mc_ll_mean.txt', delimiter=',')
    mc_ll_error = np.loadtxt('res/kink_mc_ll_error.txt', delimiter=',')

    alphas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    plt.figure(figsize=figsize(1))
    for i, alpha in enumerate(alphas):
        plt.plot(np.cumsum(mm_se_mean[:, i]), color=tableau20[2*i], label=str(alpha))
    plt.legend()
    plt.savefig('/tmp/test1.pdf', bbox_inches='tight')

    plt.figure(figsize=figsize(1))
    for i, alpha in enumerate(alphas):
        plt.plot(np.cumsum(mc_se_mean[:, i]), color=tableau20[2*i], label=str(alpha))
    plt.legend()
    plt.savefig('/tmp/test2.pdf', bbox_inches='tight')

    plt.figure(figsize=figsize(1))
    for i, alpha in enumerate(alphas):
        plt.plot(np.cumsum(mm_ll_mean[:, i]), color=tableau20[2*i], label=str(alpha))
    plt.legend()
    plt.savefig('/tmp/test3.pdf', bbox_inches='tight')

    plt.figure(figsize=figsize(1))
    for i, alpha in enumerate(alphas):
        plt.plot(np.cumsum(mc_ll_mean[:, i]), color=tableau20[2*i], label=str(alpha))
    plt.legend()
    plt.savefig('/tmp/test4.pdf', bbox_inches='tight')


if __name__ == '__main__':
    # predict_using_trained_models()
    plot_res()