import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../../../')
import numpy as np
import scipy.stats
import os, sys
import geepee.pep_models as pep
import geepee.aep_models as aep
from geepee.config import PROP_MC, PROP_MM
import pdb

import matplotlib as mpl
import matplotlib.gridspec as gridspec
# mpl.use('pgf')

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

np.random.seed(0)

# We first define several utility functions

def kink_true(x):
    fx = np.zeros(x.shape)
    for t in range(x.shape[0]):
        xt = x[t]
        if xt < 4:
            fx[t] = xt + 1
        else:
            fx[t] = -4*xt + 21
    return fx


def kink(T, process_noise, obs_noise, xprev=None):
    if xprev is None:
        xprev = np.random.randn()
    y = np.zeros([T, ])
    x = np.zeros([T, ])
    xtrue = np.zeros([T, ])
    for t in range(T):
        if xprev < 4:
            fx = xprev + 1
        else:
            fx = -4*xprev + 21

        xtrue[t] = fx
        x[t] = fx + np.sqrt(process_noise)*np.random.randn()
        xprev = x[t]
        y[t] = x[t] + np.sqrt(obs_noise)*np.random.randn()

    return xtrue, x, y

def plot_latent(model, y, plot_title='', color='k'):
    # make prediction on some test inputs
    N_test = 200
    hypers = model.get_hypers()
    C = model.get_hypers()['C_emission'][0, 0]
    x_test = np.linspace(-5, 7, N_test) / C
    x_test = np.reshape(x_test, [N_test, 1])
    zu = model.dyn_layer.zu
    mu, vu = model.predict_f(zu)
    mf, vf = model.predict_f(x_test)
    my, vy = model.predict_y(x_test)
    # plot function
    fig = plt.figure(figsize=figsize(1))
    ax = fig.add_subplot(111)
    ax.plot(C*x_test[:,0], my[:,0], '-', color=color)
    ax.fill_between(
        C*x_test[:,0], 
        my[:,0] + 2*np.sqrt(vy[:, 0]), 
        my[:,0] - 2*np.sqrt(vy[:, 0]), 
        alpha=0.3, edgecolor=color, facecolor=color)
    ax.plot(
        y[0:model.N-1], 
        y[1:model.N], 
        'k+', alpha=0.5)
    text = 'ls=%.3f, sf=%.3f, sx=%.3f, c=%.3f, sy=%.3f' % (
        np.exp(hypers['ls_dynamic'][0]),
        np.exp(hypers['sf_dynamic'][0]),
        np.exp(hypers['sn'][0]),
        hypers['C_emission'][0, 0],
        np.exp(hypers['R_emission'][0]))
    ax.plot(C*x_test[:,0], kink_true(C*x_test[:,0]), '-', color='k')
    mx, vx = model.get_posterior_x()
    ax.set_xlabel(r'$y_{t-1}$')
    ax.set_ylabel(r'$y_{t}$')
    ax.set_xlim([-5, 7])
    ax.set_ylim([-6, 6])
    ax.text(0.22, 0.03, text, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    plt.savefig('/tmp/pep_correction_fixed_hyper_kink_'+plot_title+'.pdf', bbox_inches='tight')

def run_pep():
    # generate a dataset from the kink function above
    T = 150
    process_noise = 0.2
    obs_noise = 0.05
    (xtrue, x, y) = kink(T, process_noise, obs_noise)
    y_train = y[:T]
    y_train = np.reshape(y_train, [y_train.shape[0], 1])
    Dlatent = 1
    Dobs = 1
    M = 20

    alphas = [0.0001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0]
    for i, alpha in enumerate(alphas):
        # for prop_mode in [PROP_MM, PROP_MC]:
        for prop_mode in [PROP_MM]:
            print 'alpha = %.6f, prop_mode %s' % (alpha, prop_mode)
            # create AEP model
            np.random.seed(42)
            model_aep = aep.SGPSSM(y_train, Dlatent, M, 
                lik='Gaussian', prior_mean=0, prior_var=1)
            model_aep.load_model('../compare_energies/tmp/compare_energies_kink_model_aep_%.4f_%s.pickle' % (alpha, prop_mode))
            opt_hypers = model_aep.get_hypers()
            # plot_latent(model_aep, y_train, 'PEP_wtf_%.4f_%s' % (alpha, prop_mode), color=tableau20[2 + i*2])

            model_pep = pep.SGPSSM(y_train, Dlatent, M, 
                lik='Gaussian', prior_mean=0, prior_var=1)
            model_pep.update_hypers(opt_hypers)
            aep_sgp_layer = model_aep.dyn_layer
            Nm1 = aep_sgp_layer.N
            model_pep.dyn_layer.t1 = 1.0/Nm1 * np.tile(
                aep_sgp_layer.theta_2[np.newaxis, :, :], [Nm1, 1, 1])
            model_pep.dyn_layer.t2 = 1.0/Nm1 * np.tile(
                aep_sgp_layer.theta_1[np.newaxis, :, :, :], [Nm1, 1, 1, 1])
            model_pep.dyn_layer.update_posterior()
            model_pep.x_prev_1 = np.copy(model_aep.x_factor_1)
            model_pep.x_prev_2 = np.copy(model_aep.x_factor_2)
            model_pep.x_next_1 = np.copy(model_aep.x_factor_1)
            model_pep.x_next_2 = np.copy(model_aep.x_factor_2)
            model_pep.x_up_1 = np.copy(model_aep.x_factor_1)
            model_pep.x_up_2 = np.copy(model_aep.x_factor_2)
            model_pep.x_prev_1[0, :] = 0
            model_pep.x_prev_2[0, :] = 0
            model_pep.x_next_1[-1, :] = 0
            model_pep.x_next_2[-1, :] = 0
            # run EP
            decay = 0.95
            parallel = True
            no_epochs = 2000
            model_pep.inference(no_epochs=no_epochs, alpha=alpha, parallel=parallel, decay=decay)
            plot_latent(model_pep, y_train, 'PEP_%.4f_%s' % (alpha, prop_mode), color=tableau20[2 + i*2])


if __name__ == '__main__':
    run_pep()
