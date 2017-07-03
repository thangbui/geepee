import itertools
import numpy as np
import matplotlib as mpl
import scipy.stats
from utils import find_rank
mpl.use('pgf')

np.random.seed(10)

def figsize(scale):
    fig_width_pt = 433.62                           # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.5            # Aesthetic ratio (you could change this)
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
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        r"\usepackage{amsmath}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)


import matplotlib.pyplot as plt
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

def savefig(filename):
    # plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))
    # plt.savefig('{}.eps'.format(filename))


datasets = ['australian', 'breast', 'crabs', 'iono', 'pima','sonar']
Ms = [5, 10, 20, 50, 100]

alphas = [0, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
no_trials = 20
pep_res_path = '/scratch/tdb40/geepee_gpc_pep_results/'
vfe_res_path = '/scratch/tdb40/geepee_gpc_vfe_results/'


smse_pep_res = np.zeros((len(datasets), len(alphas), len(Ms), no_trials))
for i, dataset in enumerate(datasets):
    print dataset
    for j, M in enumerate(Ms):
        for k, alpha in enumerate(alphas):
            if alpha == 0:
                fname = vfe_res_path + dataset + '_vfe_' + str(M) + '.error'
            else:
                fname = pep_res_path + dataset + '_pep_' + str(M) + '_' + str(alpha) + '.error'
            d1 = np.array(np.loadtxt(fname), ndmin=1)
            print dataset, M, alpha
            smse_pep_res[i, k, j, :] = d1


smll_pep_res = np.zeros((len(datasets), len(alphas), len(Ms), no_trials))
for i, dataset in enumerate(datasets):
    for j, M in enumerate(Ms):
        for k, alpha in enumerate(alphas):
            if alpha == 0:
                fname = vfe_res_path + dataset + '_vfe_' + str(M) + '.nll'
            else:
                fname = pep_res_path + dataset + '_pep_' + str(M) + '_' + str(alpha) + '.nll'
            d1 = np.array(np.loadtxt(fname), ndmin=1)
            
            smll_pep_res[i, k, j, :] = d1

no_alphas = len(alphas)

smse_alpha = smse_pep_res.transpose((0, 2, 3, 1))
smse_alpha = smse_alpha.reshape((len(datasets)*len(Ms)*no_trials, no_alphas))
ranks = np.empty((smse_alpha.shape[0], no_alphas), int)
for i in range(smse_alpha.shape[0]):
    res_all_i = smse_alpha[i, :]
    #temp_arg = res_all_i.argsort()
    #ranks[i, temp_arg] = np.arange(no_alphas)
    ranks[i, :] = find_rank(res_all_i)

smse_ranks_mean = np.mean(ranks, axis=0)
smse_ranks_error = np.std(ranks, axis=0) / np.sqrt(ranks.shape[0])

smll_alpha = smll_pep_res.transpose((0, 2, 3, 1))
smll_alpha = smll_alpha.reshape((len(datasets)*len(Ms)*no_trials, no_alphas))
ranks = np.empty((smll_alpha.shape[0], no_alphas), int)
for i in range(smll_alpha.shape[0]):
    res_all_i = smll_alpha[i, :]
    #temp_arg = res_all_i.argsort()
    #ranks[i, temp_arg] = np.arange(no_alphas)
    ranks[i, :] = find_rank(res_all_i)

smll_ranks_mean = np.mean(ranks, axis=0)
smll_ranks_error = np.std(ranks, axis=0) / np.sqrt(ranks.shape[0])

fig, axs = plt.subplots(1, 2, figsize=figsize(1), sharey=False, sharex=False)

axs[0].errorbar(alphas, smse_ranks_mean, yerr=2*smse_ranks_error, linewidth=2, color='k')
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel('Error rank')
axs[0].set_xlim([-0.02, 1.02])
# axs[0].set_xlim([-0.02, 0.1])
axs[1].errorbar(alphas, smll_ranks_mean, yerr=2*smll_ranks_error, linewidth=2, color='k')
axs[1].set_xlabel(r'$\alpha$')
axs[1].set_ylabel('NLL rank')
axs[1].set_xlim([-0.02, 1.02])
# axs[1].set_xlim([-0.02, 0.1])
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.15)
savefig('figs/cla_rank_plot')
