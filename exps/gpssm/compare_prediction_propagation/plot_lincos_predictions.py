import sys
sys.path.append('../../../')
import numpy as np
import scipy.stats
import os, sys
import geepee.aep_models as aep
from geepee.config import PROP_MC, PROP_MM, PROP_LIN
import pdb
from scipy.misc import logsumexp
np.random.seed(0)

import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 488.13                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/1.8            # Aesthetic ratio (you could change this)
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
   
alphas = np.array([0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1])
M = 20
np.random.seed(0)
k = np.random.randint(0, 20)
# k = 15
alpha = 0.5
prop_mode = PROP_MM
color_loc = int(2*np.where(alphas==alpha)[0])
T_test = 20
T_train = 200

dyn_noises = [0.2]
# emi_noises = [0.05, 0.2]
emi_noises = [0.2]
for dyn_noise in dyn_noises:
    for emi_noise in emi_noises:

        y_train = np.loadtxt('data/lincos_train_%d_%.2f_%.2f.txt'%(k, dyn_noise, emi_noise))
        y_train = np.reshape(y_train, [y_train.shape[0], 1])
        y_test = np.loadtxt('data/lincos_test_%d_%.2f_%.2f.txt'%(k, dyn_noise, emi_noise))
        y_test = np.reshape(y_test, [y_test.shape[0], 1])

        print 'k %d, alpha = %.6f, prop_mode %s' % (k, alpha, prop_mode)
        model_aep = aep.SGPSSM(y_train, 1, M, 
            lik='Gaussian', prior_mean=0, prior_var=1)
        model_aep.load_model('trained_models/lincos_aep_model_index_%d_%.2f_%.2f_M_%d_alpha_%.4f_%s.pickle' % (k, dyn_noise, emi_noise, M, alpha, prop_mode))

        # predict using MM and MC, TODO: lin
        _, _, my_MM, _, vy_MM = model_aep.predict_forward(T_test, prop_mode=PROP_MM)
        _, my_MC, vy_MC = model_aep.predict_forward(T_test, prop_mode=PROP_MC, no_samples=500)
        _, _, my_LIN, _, vy_LIN = model_aep.predict_forward(T_test, prop_mode=PROP_LIN)

        my_MM = my_MM[:, 0]
        vy_MM = vy_MM[:, 0]

        my_LIN = my_LIN[:, 0]
        vy_LIN = vy_LIN[:, 0]

        my_MC = my_MC[:, :, 0].T
        vy_MC = vy_MC[:, :, 0].T

        y_test = y_test[:, 0]

        no_plot_samples = 30
        t_train = np.arange(T_train) + 1
        t_test = np.arange(T_train, T_train + T_test) + 1

        # plot data and predictions
        fig, axs = plt.subplots(4, 1, figsize=figsize(1))
        # plot full data
        axs[0].plot(t_train, y_train[:, 0], 'k-')
        axs[0].plot(t_test, y_test, 'k*-', linewidth=2)
        axs[0].set_xlim([0, T_train + T_test + 1])
        axs[0].set_ylim([-10, 10])

        # plot MM prediction
        axs[1].plot(t_train, y_train[:, 0], 'k-')
        axs[1].plot(t_test, my_MM, '-', color=tableau20[color_loc], linewidth=1.5)
        axs[1].fill_between(
            t_test, 
            my_MM + 2*np.sqrt(vy_MM),
            my_MM - 2*np.sqrt(vy_MM),
            alpha=0.3, edgecolor=tableau20[color_loc], facecolor=tableau20[color_loc])
        axs[1].plot(t_test, y_test, 'k*-', linewidth=2)
        axs[1].set_ylim([-10, 10])
        axs[1].set_xlim([T_train - 10 + 0.5, T_train + T_test + 0.5])

        # plot LIN prediction TODO
        axs[2].plot(t_train, y_train[:, 0], 'k-')
        axs[2].plot(t_test, my_LIN, '-', color=tableau20[color_loc], linewidth=1.5)
        axs[2].fill_between(
            t_test, 
            my_LIN + 2*np.sqrt(vy_LIN),
            my_LIN - 2*np.sqrt(vy_LIN),
            alpha=0.3, edgecolor=tableau20[color_loc], facecolor=tableau20[color_loc])
        axs[2].plot(t_test, y_test, 'k*-', linewidth=2)
        axs[2].set_ylim([-10, 10])
        axs[2].set_xlim([T_train - 10 + 0.5, T_train + T_test + 0.5])

        # plot MC prediction
        axs[3].plot(t_train, y_train[:, 0], 'k-')
        for n in range(no_plot_samples):
            axs[3].plot(
                t_test, my_MC[n, :], 
                '-', color=tableau20[color_loc], linewidth=0.5)
        axs[3].plot(t_test, y_test, 'k*-', linewidth=2)
        axs[3].set_ylim([-10, 10])
        axs[3].set_xlim([T_train - 10 + 0.5, T_train + T_test + 0.5])

        axs[3].set_xlabel('time step')
        axs[0].set_ylabel('y')
        axs[1].set_ylabel('y, MM')
        axs[2].set_ylabel('y, LIN')
        axs[3].set_ylabel('y, MC')

        plt.savefig('/tmp/lincos_pred_%.2f_%.2f.pdf'%(dyn_noise, emi_noise), bbox_inches='tight')



        def figsize(scale):
            fig_width_pt = 488.13                          # Get this from LaTeX using \the\textwidth
            inches_per_pt = 1.0/72.27                       # Convert pt to inch
            golden_mean = (np.sqrt(5.0)-1.0)/1.5            # Aesthetic ratio (you could change this)
            fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
            fig_height = fig_width*golden_mean              # height in inches
            fig_size = [fig_width,fig_height]
            return fig_size

        # plot marginals
        alpha_plot = np.array([0.001, 0.2, 0.5, 0.8])
        step_plot = np.array([1, 2, 5, 10, 20]) - 1
        my_MM = np.zeros([len(alpha_plot), T_test])
        vy_MM = np.zeros([len(alpha_plot), T_test])
        my_LIN = np.zeros([len(alpha_plot), T_test])
        vy_LIN = np.zeros([len(alpha_plot), T_test])
        my_MC = np.zeros([len(alpha_plot), 500, T_test])
        vy_MC = np.zeros([len(alpha_plot), 500, T_test])

        for i, alpha in enumerate(alpha_plot):
            print 'k %d, alpha = %.6f, prop_mode %s' % (k, alpha, prop_mode)
            model_aep = aep.SGPSSM(y_train, 1, M, 
                lik='Gaussian', prior_mean=0, prior_var=1)
            model_aep.load_model('trained_models/lincos_aep_model_index_%d_%.2f_%.2f_M_%d_alpha_%.4f_%s.pickle' % (k, dyn_noise, emi_noise, M, alpha, prop_mode))

            # predict using MM, LIN and MC
            _, _, my_MM_i, _, vy_MM_i = model_aep.predict_forward(T_test, prop_mode=PROP_MM)
            _, my_MC_i, vy_MC_i = model_aep.predict_forward(T_test, prop_mode=PROP_MC, no_samples=500)
            _, _, my_LIN_i, _, vy_LIN_i = model_aep.predict_forward(T_test, prop_mode=PROP_LIN)
            
            my_MM[i, :] = my_MM_i[:, 0]
            vy_MM[i, :] = vy_MM_i[:, 0]

            my_LIN[i, :] = my_LIN_i[:, 0]
            vy_LIN[i, :] = vy_LIN_i[:, 0]

            my_MC[i, :, :] = my_MC_i[:, :, 0].T
            vy_MC[i, :, :] = vy_MC_i[:, :, 0].T

        yplot = np.linspace(-10, 10, 300).reshape((300, ))
        fig, axs = plt.subplots(alpha_plot.shape[0], step_plot.shape[0], figsize=figsize(1), sharex=True)
        lines = []
        labels = []
        for i, alpha in enumerate(alpha_plot):
            color_loc = int(2*np.where(alphas==alpha)[0])
            for j, step in enumerate(step_plot):
                print i, j
                ax = axs[i, j]

                # plot test point
                ax.axvline(y_test[step], color='k', linewidth=2)

                # plot mm marginal
                pdfs = mlab.normpdf(yplot, my_MM[i, step], np.sqrt(vy_MM[i, step]))
                ax.plot(yplot, pdfs, '--', color=tableau20[color_loc], linewidth=1)
                
                # plot lin marginal
                pdfs = mlab.normpdf(yplot, my_LIN[i, step], np.sqrt(vy_LIN[i, step]))
                ax.plot(yplot, pdfs, '-.', color=tableau20[color_loc], linewidth=1)

                # plot mc marginal
                pdfs = np.zeros(yplot.shape[0])
                for n in range(500):
                    pdfs += mlab.normpdf(yplot, my_MC[i, n, step], np.sqrt(vy_MC[i, n, step]))
                
                if j == 0:
                    label = r'$\alpha=%.3f$'%alpha
                    line = ax.plot(yplot, pdfs / 500, '-', color=tableau20[color_loc], linewidth=1, label=label)[0]
                    lines.append(line)
                    labels.append(label)
                else:
                    ax.plot(yplot, pdfs / 500, '-', color=tableau20[color_loc], linewidth=1)
                
                ax.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    labelleft='off') # labels along the bottom edge are off
                ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='on',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='on') # labels along the bottom edge are off

                if i == len(alpha_plot) - 1:
                    ax.set_xlabel(r'y, step %d'%(step+1))
                    # ax.set_xticks([-2, 0, 2, 4, 6])

                if j == 0:
                    ax.set_ylabel(r'p(y), $\alpha=%.3f$'%alpha)

        mmArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='--')
        linArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='-.')
        mcArtist = plt.Line2D((0,1),(0,0), color='k')

        #Create legend from custom artist/label lists
        # fig.legend([handle for i,handle in enumerate(lines)]+[mmArtist,linArtist,mcArtist],
        #           [label for i,label in enumerate(labels)]+['MM', 'LIN', 'MC'],
        #           loc='upper center', ncol=2, handlelength=4)
        fig.legend([mmArtist,linArtist,mcArtist],
                  ['MM', 'LIN', 'MC'],
                  loc='upper center', ncol=3, handlelength=4)

        plt.savefig('/tmp/lincos_pred_marginals_%.2f_%.2f.pdf'%(dyn_noise, emi_noise), bbox_inches='tight')


