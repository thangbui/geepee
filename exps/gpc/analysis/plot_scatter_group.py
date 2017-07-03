import itertools
import numpy as np
import matplotlib as mpl
import scipy.stats
mpl.use('pgf')

np.random.seed(10)

def figsize(scale):
    # fig_width_pt = 614.295                           # Get this from LaTeX using \the\textwidth
    fig_width_pt = 1100                           # Get this from LaTeX using \the\textwidth
    fig_width_pt = 600
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2            # Aesthetic ratio (you could change this)
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
Ms = [5, 10, 20, 50, 100]
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
# cm = plt.cm.get_cmap('RdYlBu_r')
cm = plt.cm.get_cmap('viridis')

from mpl_toolkits.axes_grid.inset_locator import inset_axes
plt.style.use('ggplot')
fig, axs = plt.subplots(2, len(xaxes), figsize=figsize(1), sharey=False, sharex=False)
plt.hold(True)
for a in range(len(xaxes)):
    print a
    xa = xaxes[a]
    ya = yaxes[a]

    xvals = error_pep_res[:, xa, :, :].flatten()
    yvals = error_pep_res[:, ya, :, :].flatten()

    diff = xvals - yvals

    count_neg = np.sum(diff < 0)
    count_pos = np.sum(diff > 0)
    count_zero = np.sum(diff == 0)
    print count_pos, count_neg


    diff_std = np.std(diff)

    xvals = xvals[abs(diff) < 10*diff_std]
    yvals = yvals[abs(diff) < 10*diff_std]

    min_x = np.min([np.min(xvals), np.min(yvals)])
    max_x = np.max([np.max(xvals), np.max(yvals)])
    min_x = min_x - 0.02
    max_x = max_x
    axs[0, a].plot([min_x, max_x], [min_x, max_x], '-k', linewidth=1)
    delta_x = 1.0/7 * (max_x - min_x)
    delta_diff = delta_x / np.sqrt(2)
    axs[0, a].plot([min_x-delta_x, max_x-delta_x], [min_x, max_x], '--r', linewidth=1, zorder=0)    
    axs[0, a].plot([min_x, max_x], [min_x-delta_x, max_x-delta_x], '--b', linewidth=1, zorder=0)
    axs[0, a].scatter(xvals, yvals, color=tableau20[2*a], alpha=.7, s=4)

    inset = inset_axes(axs[0, a],
                width="40%",
                height="40%",
                loc=2)
    n, bins, patches = inset.hist(xvals-yvals, 20, normed=1, histtype='stepfilled', color=tableau20[2*a], linewidth=0.2)
    # delete
    inset.remove()
    inset = inset_axes(axs[0, a],
                width="40%",
                height="40%",
                loc=2)
    # search for bin that has 0
    loc = np.where(bins > 0)[0][0]
    bins[loc] = 0
    bins = np.delete(bins, loc-1)
    n, bins, patches = inset.hist(xvals-yvals, bins, normed=1, histtype='stepfilled', color=tableau20[2*a], linewidth=0.2) 
    
    maxabs = np.max(abs(xvals-yvals))
    inset.axvline(0, color='k', zorder=0)
    inset.axhline(0, color='k')
    inset.axvline(delta_diff, linestyle='dashed', color='b', zorder=0)
    inset.axvline(-delta_diff, linestyle='dashed', color='r', zorder=0)
    inset.set_yticks([])
    inset.set_xticks([0])
    inset.set_xlim([-maxabs, maxabs])
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    inset.spines['left'].set_visible(False)
    inset.spines['bottom'].set_linewidth(0.5)
    plt.text(0.8, 0.5,'%.0f%%' % np.round(100.0*count_pos/(count_zero+count_pos+count_neg)),
        horizontalalignment='left',
        verticalalignment='center',
        transform = inset.transAxes)
    plt.text(0.3, 0.5,'%.0f%%' % np.round(100.0*count_neg/(count_zero+count_pos+count_neg)),
        horizontalalignment='right',
        verticalalignment='center',
        transform = inset.transAxes)
    # plt.text(0.9, 0.5,'%d' % count_pos,
    #     horizontalalignment='left',
    #     verticalalignment='center',
    #     transform = inset.transAxes)
    # plt.text(0.2, 0.5,'%d' % count_neg,
    #     horizontalalignment='right',
    #     verticalalignment='center',
    #     transform = inset.transAxes)
    # if count_zero > 0:
    #     plt.text(0.5, 0.5,'%d' % count_zero,
    #         horizontalalignment='center',
    #         verticalalignment='center',
    #         transform = inset.transAxes)
    # bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # # scale values to interval [0,1]
    # col = bin_centers + maxabs
    # col /= 2.0*maxabs
    # cm = plt.cm.get_cmap('RdYlBu_r')
    # for color, patch in zip(col, patches):
    #     plt.setp(patch, 'facecolor', cm(color))
    inset.patch.set_alpha(0) 

    axs[0, a].set_xlabel('%s' % names[xa])
    axs[0, a].set_ylabel('%s' % names[ya])

    axs[0, a].set_xlim([min_x, max_x])
    axs[0, a].set_ylim([min_x, max_x])

    axs[0, a].get_yaxis().get_major_formatter().set_powerlimits((0, 0))
    axs[0, a].get_xaxis().get_major_formatter().set_powerlimits((0, 0))
    axs[0, a].set_aspect('equal', adjustable='box')
    axs[0, a].xaxis.set_ticks_position('none')
    axs[0, a].yaxis.set_ticks_position('none')
    axs[0, a].set_yticks([])
    axs[0, a].set_xticks([])
    axs[0, a].set_title('Error')



    xvals = nll_pep_res[:, xa, :, :].flatten()
    yvals = nll_pep_res[:, ya, :, :].flatten()
    diff = xvals - yvals
    count_neg = np.sum(diff < 0)
    count_pos = np.sum(diff > 0)
    count_zero = np.sum(diff == 0)
    print count_pos, count_neg
    diff_std = np.std(diff)

    xvals = xvals[abs(diff) < 10*diff_std]
    yvals = yvals[abs(diff) < 10*diff_std]

    min_x = np.min([np.min(xvals), np.min(yvals)])
    max_x = np.max([np.max(xvals), np.max(yvals)])
    min_x = min_x
    max_x = max_x
    axs[1, a].plot([min_x, max_x], [min_x, max_x], '-k', linewidth=1)
    delta_x = 1.0/7 * (max_x - min_x)
    delta_diff = delta_x / np.sqrt(2)
    axs[1, a].plot([min_x-delta_x, max_x-delta_x], [min_x, max_x], '--r', linewidth=1, zorder=0)    
    axs[1, a].plot([min_x, max_x], [min_x-delta_x, max_x-delta_x], '--b', linewidth=1, zorder=0)
    axs[1, a].scatter(xvals, yvals, color=tableau20[2*a], alpha=.7, s=4)

    inset = inset_axes(axs[1, a],
                width="40%",
                height="40%",
                loc=2)
    n, bins, patches = inset.hist(xvals-yvals, 20, normed=1, histtype='stepfilled', color=tableau20[2*a], linewidth=0.2)
    # delete
    inset.remove()
    inset = inset_axes(axs[1, a],
                width="40%",
                height="40%",
                loc=2)
    # search for bin that has 0
    loc = np.where(bins > 0)[0][0]
    bins[loc] = 0
    bins = np.delete(bins, loc-1)
    n, bins, patches = inset.hist(xvals-yvals, bins, normed=1, histtype='stepfilled', color=tableau20[2*a], linewidth=0.2) 
    inset.axvline(0, color='k', zorder=0)
    inset.axhline(0, color='k')
    inset.axvline(delta_diff, linestyle='dashed', color='b', zorder=0)
    inset.axvline(-delta_diff, linestyle='dashed', color='r', zorder=0)
    maxabs = np.max(abs(xvals-yvals))
    inset.set_yticks([])
    inset.set_xticks([0])
    inset.set_xlim([-maxabs, maxabs])
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    inset.spines['left'].set_visible(False)
    inset.spines['bottom'].set_linewidth(0.5)
    # bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # # scale values to interval [0,1]
    # col = bin_centers + maxabs
    # col /= 2.0*maxabs
    # cm = plt.cm.get_cmap('RdYlBu_r')
    # for color, patch in zip(col, patches):
    #     plt.setp(patch, 'facecolor', cm(color))
    inset.patch.set_alpha(0)
    plt.text(0.8, 0.5,'%.0f%%' % np.round(100.0*count_pos/(count_zero+count_pos+count_neg)),
        horizontalalignment='left',
        verticalalignment='center',
        transform = inset.transAxes)
    plt.text(0.3, 0.5,'%.0f%%' % np.round(100.0*count_neg/(count_zero+count_pos+count_neg)),
        horizontalalignment='right',
        verticalalignment='center',
        transform = inset.transAxes)
    # plt.text(0.9, 0.5,'%d' % count_pos,
    #     horizontalalignment='left',
    #     verticalalignment='center',
    #     transform = inset.transAxes)
    # plt.text(0.2, 0.5,'%d' % count_neg,
    #     horizontalalignment='right',
    #     verticalalignment='center',
    #     transform = inset.transAxes)
    # if count_zero > 0:
    #     plt.text(0.5, 0.5,'%d' % count_zero,
    #         horizontalalignment='center',
    #         verticalalignment='center',
    #         transform = inset.transAxes)
    axs[1, a].set_xlabel('%s' % names[xa])
    axs[1, a].set_ylabel('%s' % names[ya])

    axs[1, a].set_xlim([min_x, max_x])
    axs[1, a].set_ylim([min_x, max_x])

    axs[1, a].get_yaxis().get_major_formatter().set_powerlimits((0, 0))
    axs[1, a].get_xaxis().get_major_formatter().set_powerlimits((0, 0))
    axs[1, a].set_aspect('equal', adjustable='box')
    axs[1, a].xaxis.set_ticks_position('none')
    axs[1, a].yaxis.set_ticks_position('none')
    axs[1, a].set_yticks([])
    axs[1, a].set_xticks([])
    axs[1, a].set_title('NLL')

plt.subplots_adjust(left=0.02, bottom=0.05, right=1.0, top=0.95, wspace=0.20, hspace=0.24)
savefig('figs/cla_scatter_all')