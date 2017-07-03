import itertools
import numpy as np
import matplotlib as mpl
import scipy.stats
import pdb
mpl.use('pgf')

np.random.seed(10)

def figsize(scale):
    # fig_width_pt = 614.295                           # Get this from LaTeX using \the\textwidth
    fig_width_pt = 1100                           # Get this from LaTeX using \the\textwidth
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
    "axes.labelsize": 11,               # LaTeX default is 10pt font.
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
Ms = [5, 10, 50, 100]
alphas = [0, 0.5, 1.0]
no_trials = 20
pep_res_path = '/scratch/tdb40/geepee_gpc_pep_results/'
vfe_res_path = '/scratch/tdb40/geepee_gpc_vfe_results/'

alphas_str = ['VFE', '0.5', 'EP']


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



xaxes = [0, 0, 2]
yaxes = [1, 2, 1]
names = ['VFE', '0.5', 'EP']
cm = plt.cm.get_cmap('viridis')

from mpl_toolkits.axes_grid.inset_locator import inset_axes


for a in range(len(xaxes)):
    print a
    xa = xaxes[a]
    ya = yaxes[a]

    fig, axs = plt.subplots(4, len(datasets), figsize=figsize(1), sharey=False, sharex=False)
    plt.hold(True)

    # dataset index
    for i, dataset in enumerate(datasets):
        xvals = error_pep_res[i, xa, :, :].flatten()
        yvals = error_pep_res[i, ya, :, :].flatten()
        diff = xvals - yvals
        count_neg = np.sum(diff < 0)
        count_pos = np.sum(diff > 0)
        count_zero = np.sum(diff==0)
        diff_std = np.std(diff)
        xvals = xvals[abs(diff) < 3.5*diff_std]
        yvals = yvals[abs(diff) < 3.5*diff_std]

        min_x = np.min([np.min(xvals), np.min(yvals)])
        max_x = np.max([np.max(xvals), np.max(yvals)])
        axs[0, i].plot([min_x, max_x], [min_x, max_x], '-k', linewidth=2)
        for m, M in enumerate(Ms):
            xvals_m = error_pep_res[i, xa, m, :]
            yvals_m = error_pep_res[i, ya, m, :]
            cm = plt.cm.get_cmap('viridis')
            axs[0, i].scatter(xvals_m, yvals_m, color=cm((np.log(300)-np.log(M))*1.0/np.log(300)), alpha=.6, s=4)

        
        axs[0, i].set_xlabel('%s' % names[xa])
        axs[0, 0].set_ylabel('%s' % names[ya])

        axs[0, i].set_xlim([min_x, max_x])
        axs[0, i].set_ylim([min_x, max_x])

        axs[0, i].get_yaxis().get_major_formatter().set_powerlimits((0, 0))
        axs[0, i].get_xaxis().get_major_formatter().set_powerlimits((0, 0))
        axs[0, i].set_aspect('equal', adjustable='box')
        axs[0, i].xaxis.set_ticks_position('none')
        axs[0, i].yaxis.set_ticks_position('none')

        axs[0, i].set_yticks([])
        axs[0, i].set_xticks([])

        # inset = inset_axes(axs[0, i],
        #             width="45%", # width = 30% of parent_bbox
        #             height=0.55, # height : 1 inch
        #             loc=4)
        inset = inset_axes(axs[1, i],
                width="40%",
                height="40%",
                loc=2)
        n, bins, patches = inset.hist(xvals-yvals, 15, normed=1, linewidth=0.2)
        # delete
        inset.remove()
        inset = axs[1, i]
        # if a == 2:
        #     pdb.set_trace()
        # search for bin that has 0
        loc = np.where(bins >= 0)[0][0]
        bins[loc] = 0
        # print a, dataset, bins
        if loc != 1:
            bins = np.delete(bins, loc-1)
        n, bins, patches = inset.hist(xvals-yvals, bins, normed=1, linewidth=0.2)
        inset.axvline(0, color='k')
        maxabs = np.max(abs(xvals-yvals))
        plt.text(0.8, 0.6,'%d' % count_pos,
            horizontalalignment='left',
            verticalalignment='center',
            transform = inset.transAxes)
        plt.text(0.2, 0.6,'%d' % count_neg,
            horizontalalignment='right',
            verticalalignment='center',
            transform = inset.transAxes)
        if count_zero > 0:
            plt.text(0.5, 0.6,'%d' % count_zero,
                horizontalalignment='center',
                verticalalignment='center',
                transform = inset.transAxes)
        inset.set_yticks([])
        inset.set_xticks([0])
        inset.set_xlim([-maxabs, maxabs])
        inset.spines['top'].set_visible(False)
        inset.spines['right'].set_visible(False)
        inset.spines['left'].set_visible(False)
        inset.spines['bottom'].set_linewidth(0.5)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # scale values to interval [0,1]
        col = bin_centers + maxabs
        col /= 2.0*maxabs
        cm = plt.cm.get_cmap('RdYlBu_r')
        for color, patch in zip(col, patches):
            plt.setp(patch, 'facecolor', cm(color))
        inset.patch.set_alpha(0)
        # inset.set_xlabel('%s ' % dataset) 
  
    # dataset index
    for i, dataset in enumerate(datasets):
        xvals = nll_pep_res[i, xa, :, :].flatten()
        yvals = nll_pep_res[i, ya, :, :].flatten()
        diff = xvals - yvals
        count_neg = np.sum(diff < 0)
        count_pos = np.sum(diff > 0)
        count_zero = np.sum(diff==0)
        diff_std = np.std(diff)
        # xvals = xvals[abs(diff) < 3.5*diff_std]
        # yvals = yvals[abs(diff) < 3.5*diff_std]

        min_x = np.min([np.min(xvals), np.min(yvals)])
        max_x = np.max([np.max(xvals), np.max(yvals)])
        axs[2, i].plot([min_x, max_x], [min_x, max_x], '-k', linewidth=2)
        cm = plt.cm.get_cmap('viridis')
        for m, M in enumerate(Ms):
            xvals_m = nll_pep_res[i, xa, m, :]
            yvals_m = nll_pep_res[i, ya, m, :]
            axs[2, i].scatter(xvals_m, yvals_m, color=cm((np.log(300)-np.log(M))*1.0/np.log(300)), alpha=.6, s=4)         
            
        axs[2, i].set_xlabel('%s' % names[xa])
        axs[2, 0].set_ylabel('%s' % names[ya])

        axs[2, i].set_xlim([min_x, max_x])
        axs[2, i].set_ylim([min_x, max_x])

        axs[2, i].get_yaxis().get_major_formatter().set_powerlimits((0, 0))
        axs[2, i].get_xaxis().get_major_formatter().set_powerlimits((0, 0))
        axs[2, i].set_aspect('equal', adjustable='box')
        axs[2, i].xaxis.set_ticks_position('none')
        axs[2, i].yaxis.set_ticks_position('none')

        axs[2, i].set_yticks([])
        axs[2, i].set_xticks([])

        inset = inset_axes(axs[3, i],
                width="40%",
                height="40%",
                loc=2)
        n, bins, patches = inset.hist(xvals-yvals, 15, normed=1, linewidth=0.2)
        # delete
        inset.remove()
        inset = axs[3, i]
        # search for bin that has 0
        loc = np.where(bins >= 0)[0][0]
        if loc != 1:
            bins = np.delete(bins, loc-1)
        bins = np.delete(bins, loc-1)
        n, bins, patches = inset.hist(xvals-yvals, bins, normed=1, linewidth=0.2)
        inset.axvline(0, color='k')
        maxabs = np.max(abs(xvals-yvals))
        plt.text(0.8, 0.6,'%d' % count_pos,
            horizontalalignment='left',
            verticalalignment='center',
            transform = inset.transAxes)
        plt.text(0.2, 0.6,'%d' % count_neg,
            horizontalalignment='right',
            verticalalignment='center',
            transform = inset.transAxes)
        if count_zero > 0:
            plt.text(0.5, 0.6,'%d' % count_zero,
                horizontalalignment='center',
                verticalalignment='center',
                transform = inset.transAxes)
        inset.set_yticks([])
        inset.set_xticks([0])
        inset.set_xlim([-maxabs, maxabs])
        inset.spines['top'].set_visible(False)
        inset.spines['right'].set_visible(False)
        inset.spines['left'].set_visible(False)
        inset.spines['bottom'].set_linewidth(0.5)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # scale values to interval [0,1]
        col = bin_centers + maxabs
        col /= 2.0*maxabs
        cm = plt.cm.get_cmap('RdYlBu_r')
        for color, patch in zip(col, patches):
            plt.setp(patch, 'facecolor', cm(color))
        inset.patch.set_alpha(0)
        inset.set_xlabel('%s ' % dataset)   
    
    savefig('figs/cla_scatter%d'%a)