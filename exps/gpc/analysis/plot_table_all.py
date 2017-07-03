import itertools
import numpy as np
import matplotlib as mpl
import scipy.stats
mpl.use('pgf')

np.random.seed(10)

def figsize(scale):
    fig_width_pt = 650.43                           # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/1.9            # Aesthetic ratio (you could change this)
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

error_pep_res = np.zeros((len(datasets), len(alphas), len(Ms), no_trials))
for i, dataset in enumerate(datasets):
    print dataset
    for j, M in enumerate(Ms):
        for k, alpha in enumerate(alphas):
            if alpha == 0:
                fname = vfe_res_path + dataset + '_vfe_' + str(M) + '.error'
            else:
                fname = pep_res_path + dataset + '_pep_' + str(M) + '_' + str(alpha) + '.error'
            d1 = np.array(np.loadtxt(fname), ndmin=1)            
            error_pep_res[i, k, j, :] = d1

nll_pep_res = np.zeros((len(datasets), len(alphas), len(Ms), no_trials))
for i, dataset in enumerate(datasets):
    for j, M in enumerate(Ms):
        for k, alpha in enumerate(alphas):
            if alpha == 0:
                fname = vfe_res_path + dataset + '_vfe_' + str(M) + '.nll'
            else:
                fname = pep_res_path + dataset + '_pep_' + str(M) + '_' + str(alpha) + '.nll'
            d1 = np.array(np.loadtxt(fname), ndmin=1)
            
            nll_pep_res[i, k, j, :] = d1


error_pep_mean = np.mean(error_pep_res, axis=3)
error_pep_error = np.std(error_pep_res, axis=3) / np.sqrt(no_trials)
nll_pep_mean = np.mean(nll_pep_res, axis=3)
nll_pep_error = np.std(nll_pep_res, axis=3) / np.sqrt(no_trials)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


fig, axs = plt.subplots(len(datasets), len(alphas)+1, figsize=figsize(2), sharey=False, sharex=False)
axs = np.array(axs)
axs = np.reshape(axs, [len(datasets), len(alphas)+1])

# dataset index
for i, dataset in enumerate(datasets):
    x_val = np.array(Ms)
    for j, alpha in enumerate(alphas):
        axs[i, len(alphas)].plot(x_val, error_pep_mean[i, j, :], '+-', color=tableau20[j*2], linewidth=1)

    axs[i, len(alphas)].set_xlabel('M')

for i, dataset in enumerate(datasets):
    # alpha index
    min_y = 10
    max_y = -10
    for j, alpha in enumerate(alphas):
        x_val = np.array(Ms)
        for k in range(no_trials):
            smse_val = error_pep_res[i, j, :, k]
            axs[i, j].plot(x_val, smse_val, '-', color=tableau20[j*2], alpha=0.2)

        ys = axs[i, j].get_ylim()
        if min_y > ys[0]:
            min_y = ys[0]
        if max_y < ys[1]:
            max_y = ys[1]

        axs[i, j].plot(x_val, error_pep_mean[i, j, :], '-', color=tableau20[j*2], linewidth=2)
        # axs[i, j].set_ylim(axs[i, len(alphas)].get_ylim())
        
        axs[0, j].set_title('%.3f' % alpha)
        axs[len(datasets)-1, j].set_xlabel('M')
    for j in range(len(alphas)+1):
        axs[i, j].set_ylim([min_y, max_y])

    axs[i, 0].set_ylabel(dataset + '\nError')

savefig('figs/cla_error_res_table')


fig, axs = plt.subplots(len(datasets), len(alphas)+1, figsize=figsize(2), sharey=False, sharex=False)
axs = np.array(axs)
axs = np.reshape(axs, [len(datasets), len(alphas)+1])
# dataset index
for i, dataset in enumerate(datasets):
    for j, alpha in enumerate(alphas):
        x_val = np.array(Ms)
        axs[i, len(alphas)].plot(x_val, nll_pep_mean[i, j, :], '+-', color=tableau20[j*2], linewidth=1)

    # axs[i, len(datasets)-1].set_xlabel('M')

for i, dataset in enumerate(datasets):
    min_y = 10
    max_y = -10
    # alpha index
    for j, alpha in enumerate(alphas):
        x_val = np.array(Ms)
        for k in range(no_trials):
            smll_val = nll_pep_res[i, j, :, k]
            axs[i, j].plot(x_val, smll_val, '-', color=tableau20[j*2], alpha=0.2)

        ys = axs[i, j].get_ylim()
        if min_y > ys[0]:
            min_y = ys[0]
        if max_y < ys[1]:
            max_y = ys[1]

        axs[i, j].plot(x_val, nll_pep_mean[i, j, :], '-', color=tableau20[j*2], linewidth=2)
        # axs[i, j].set_ylim(axs[i, len(alphas)].get_ylim())

        axs[0, j].set_title('%.3f' % alpha)
        axs[len(datasets)-1, j].set_xlabel('M')



    axs[i, 0].set_ylabel(dataset + '\nNLL')

    for j, alpha in enumerate(alphas):
        axs[i, len(alphas)].plot(x_val, nll_pep_mean[i, j, :], color=tableau20[j*2], linewidth=1)

    # axs[i, len(alphas)].set_xlabel('M')

    for j in range(len(alphas)+1):
        axs[i, j].set_ylim([min_y, max_y])

savefig('figs/cla_nll_res_table')

