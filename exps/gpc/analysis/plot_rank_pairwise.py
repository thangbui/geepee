import itertools
import numpy as np
import matplotlib as mpl
import scipy.stats
from utils import find_rank
mpl.use('pgf')

np.random.seed(10)

def figsize(scale):
    fig_width_pt = 433.62/2                           # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 8,
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


dataset_fullname = {'australian': 'australian',
          'breast': 'breast',
          'crabs': 'crabs',
          'iono': 'iono',
          'pima': 'pima',
          'sonar': 'sonar'}

datasets = ['australian', 'breast', 'crabs', 'iono', 'pima','sonar']
Ms = [5, 10, 20, 50, 100]
pep_res_path = '/scratch/tdb40/geepee_gpc_pep_results/'
vfe_res_path = '/scratch/tdb40/geepee_gpc_vfe_results/'
fig, axs = plt.subplots(8, 2, figsize=figsize(1), sharey=False, sharex=False)
alphas_all = [0, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
alphas_str = ['VFE', 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 'EP']
plt.style.use('ggplot')
fig, axs = plt.subplots(len(alphas_all), len(alphas_all), figsize=figsize(1), sharey=False, sharex=False)
for a_ind_i, a_i in enumerate(alphas_all):
    for a_ind_j, a_j in enumerate(alphas_all):
        print a_ind_i, a_ind_j
        if a_ind_i == a_ind_j:
            print 'skipping this'
            # axs[a_ind_i, a_ind_j].axis('off')
            axs[a_ind_i, 0].set_ylabel('%s' % str(alphas_str[a_ind_i]), fontsize=7)
            axs[0, a_ind_j].set_title('%s' % str(alphas_str[a_ind_j]), fontsize=7)
            axs[a_ind_i, a_ind_j].spines['top'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['right'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['bottom'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['left'].set_visible(False)
            axs[a_ind_i, a_ind_j].xaxis.set_major_locator(plt.NullLocator())
            axs[a_ind_i, a_ind_j].yaxis.set_major_locator(plt.NullLocator())
        else:
            alphas = [a_i, a_j]
            no_trials = 20

            smse_pep_res = np.zeros((len(datasets), len(alphas), len(Ms), no_trials))
            for i, dataset in enumerate(datasets):
                for j, M in enumerate(Ms):
                    for k, alpha in enumerate(alphas):
                        alpha_str = str(alpha)
                        if alpha == 0:
                            fname = vfe_res_path + dataset_fullname[dataset] + '_vfe_' + str(M) + '.error'
                        else:    
                            fname = pep_res_path + dataset_fullname[dataset] + '_pep_' + str(M) + '_' + alpha_str + '.error'
                        d1 = np.array(np.loadtxt(fname), ndmin=1)
                        print dataset, M, alpha 
                        smse_pep_res[i, k, j, :] = d1

            no_alphas = len(alphas)

            smse_alpha = smse_pep_res.transpose((0, 2, 3, 1))
            smse_alpha = smse_alpha.reshape((len(datasets)*len(Ms)*no_trials, no_alphas))
            ranks = np.empty((smse_alpha.shape[0], no_alphas), int)
            for i in range(smse_alpha.shape[0]):
                res_all_i = smse_alpha[i, :]
                ranks[i, :] = find_rank(res_all_i)

            smse_ranks_mean = np.mean(ranks, axis=0)
            smse_ranks_error = np.std(ranks, axis=0) / np.sqrt(ranks.shape[0])

            axs[a_ind_i, a_ind_j].errorbar([0, 1], smse_ranks_mean, yerr=2*smse_ranks_error, linewidth=0.5, color='k', capsize=2)

            if smse_ranks_mean[0] < (smse_ranks_mean[1] - 3*smse_ranks_error[1]) and smse_ranks_mean[1] > (smse_ranks_mean[0] + 3*smse_ranks_error[0]):
                c = plt.Circle((0, smse_ranks_mean[0]), 0.1, color='r', fill=False)
                axs[a_ind_i, a_ind_j].add_artist(c)

            if smse_ranks_mean[1] < (smse_ranks_mean[0] - 3*smse_ranks_error[0]) and smse_ranks_mean[0] > (smse_ranks_mean[1] + 3*smse_ranks_error[1]):
                c = plt.Circle((1, smse_ranks_mean[1]), 0.1, color='b', fill=False)
                axs[a_ind_i, a_ind_j].add_artist(c)

            axs[a_ind_i, 0].set_ylabel(r'%s' % str(alphas_str[a_ind_i]), fontsize=7)
            axs[0, a_ind_j].set_title(r'%s' % str(alphas_str[a_ind_j]), fontsize=7)
            axs[a_ind_i, a_ind_j].spines['top'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['right'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['bottom'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['left'].set_visible(False)
            axs[a_ind_i, a_ind_j].xaxis.set_major_locator(plt.NullLocator())
            axs[a_ind_i, a_ind_j].yaxis.set_major_locator(plt.NullLocator())
            axs[a_ind_i, a_ind_j].set_ylim([-0.1, 1.1])
            axs[a_ind_i, a_ind_j].set_xlim([-0.15, 1.15])

plt.suptitle('Error rank', fontsize=8)
plt.subplots_adjust(wspace=0.05, hspace=0.05)            
savefig('figs/cla_rank_plot_pairwise_error')


fig, axs = plt.subplots(len(alphas_all), len(alphas_all), figsize=figsize(1), sharey=False, sharex=False)
for a_ind_i, a_i in enumerate(alphas_all):
    for a_ind_j, a_j in enumerate(alphas_all):
        print a_ind_i, a_ind_j
        if a_ind_i == a_ind_j:
            print 'skipping this'
            # axs[a_ind_i, a_ind_j].axis('off')
            axs[a_ind_i, 0].set_ylabel('%s' % str(alphas_str[a_ind_i]), fontsize=7)
            axs[0, a_ind_j].set_title('%s' % str(alphas_str[a_ind_j]), fontsize=7)
            axs[a_ind_i, a_ind_j].spines['top'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['right'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['bottom'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['left'].set_visible(False)
            axs[a_ind_i, a_ind_j].xaxis.set_major_locator(plt.NullLocator())
            axs[a_ind_i, a_ind_j].yaxis.set_major_locator(plt.NullLocator())
        else:
            alphas = [a_i, a_j]
            no_trials = 20
            res_path = '/scratch/tdb40/pepsgp_cla_results/'

            smll_pep_res = np.zeros((len(datasets), len(alphas), len(Ms), no_trials))
            for i, dataset in enumerate(datasets):
                for j, M in enumerate(Ms):
                    for k, alpha in enumerate(alphas):
                        alpha_str = str(alpha)
                        if alpha == 0:
                            fname = vfe_res_path + dataset_fullname[dataset] + '_vfe_' + str(M) + '.nll'
                        else:
                            fname = pep_res_path + dataset_fullname[dataset] + '_pep_' + str(M) + '_' + alpha_str + '.nll'
                        d1 = np.array(np.loadtxt(fname), ndmin=1)
                        
                        smll_pep_res[i, k, j, :] = d1

            no_alphas = len(alphas)

            smll_alpha = smll_pep_res.transpose((0, 2, 3, 1))
            smll_alpha = smll_alpha.reshape((len(datasets)*len(Ms)*no_trials, no_alphas))
            ranks = np.empty((smll_alpha.shape[0], no_alphas), int)
            for i in range(smll_alpha.shape[0]):
                res_all_i = smll_alpha[i, :]
                ranks[i, :] = find_rank(res_all_i)

            smll_ranks_mean = np.mean(ranks, axis=0)
            smll_ranks_error = np.std(ranks, axis=0) / np.sqrt(ranks.shape[0])


            axs[a_ind_i, a_ind_j].errorbar([0, 1], smll_ranks_mean, yerr=2*smll_ranks_error, linewidth=0.5, color='k', capsize=2)

            if smll_ranks_mean[0] < (smll_ranks_mean[1] - 3*smll_ranks_error[1]) and smll_ranks_mean[1] > (smll_ranks_mean[0] + 3*smll_ranks_error[0]):
                c = plt.Circle((0, smll_ranks_mean[0]), 0.1, color='r', fill=False)
                axs[a_ind_i, a_ind_j].add_artist(c)

            if smll_ranks_mean[1] < (smll_ranks_mean[0] - 3*smll_ranks_error[0]) and smll_ranks_mean[0] > (smll_ranks_mean[1] + 3*smll_ranks_error[1]):
                c = plt.Circle((1, smll_ranks_mean[1]), 0.1, color='b', fill=False)
                axs[a_ind_i, a_ind_j].add_artist(c)

            axs[a_ind_i, 0].set_ylabel(r'%s' % str(alphas_str[a_ind_i]), fontsize=7)
            axs[0, a_ind_j].set_title(r'%s' % str(alphas_str[a_ind_j]), fontsize=7)
            axs[a_ind_i, a_ind_j].spines['top'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['right'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['bottom'].set_visible(False)
            axs[a_ind_i, a_ind_j].spines['left'].set_visible(False)
            axs[a_ind_i, a_ind_j].xaxis.set_major_locator(plt.NullLocator())
            axs[a_ind_i, a_ind_j].yaxis.set_major_locator(plt.NullLocator())
            axs[a_ind_i, a_ind_j].set_ylim([-0.1, 1.1])
            axs[a_ind_i, a_ind_j].set_xlim([-0.15, 1.15])

plt.suptitle('NLL rank', fontsize=8)
plt.subplots_adjust(wspace=0.05, hspace=0.05)        
savefig('figs/cla_rank_plot_pairwise_nll')
