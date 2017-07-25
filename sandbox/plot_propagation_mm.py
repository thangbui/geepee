import numpy as np
import scipy.linalg as spla
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.use('pgf')
from sklearn.neighbors import KernelDensity
from scipy.stats.kde import gaussian_kde

from geepee.kernels import *

np.random.seed(100)

def figsize(scale):
    fig_width_pt = 488.13                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    ratio = 2.2
    fig_width = fig_width_pt*inches_per_pt    # width in inches
    fig_height = fig_width*scale              # height in inches
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

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[ 0 ]))

def matrixInverse(M):
    return chol2inv(spla.cholesky(M, lower=False))

# hyperparameters
lls = np.array([np.log(1)])
lsf = 0
lsn = np.log(0.02)

# draw a function
N_train = 100
x_train = np.linspace(-5, 5, N_train)
x_train = np.reshape(x_train, (N_train, 1))
Kff = compute_kernel(lls, lsf, x_train, x_train) + 1e-6 * np.eye(N_train)
f_train = np.dot(spla.cholesky(Kff).T, np.random.randn(N_train, 1))
y_train = f_train + np.exp(lsn) * np.random.randn(N_train, 1)

# use pseudo points
M = 30
z = np.linspace(-4.5, 4.5, M)
z = np.reshape(z, (M, 1))
Kuu = compute_kernel(lls, lsf, z, z) + 1e-6 * np.eye(M)
Kfu = compute_kernel(lls, lsf, x_train, z)
Kuuinv = matrixInverse(Kuu)

Qff = np.dot(Kfu, np.dot(Kuuinv, Kfu.T))
lsn = lsn
Qffplus = Qff + np.exp(lsn) * np.eye(N_train)
Qffplus_inv = matrixInverse(Qffplus)
mu = np.dot(Kfu.T, np.dot(Qffplus_inv, y_train))
Vu = Kuu - np.dot(Kfu.T, np.dot(Qffplus_inv, Kfu))

# making prediction
N_test = 400
x_test = np.linspace(-6, 7, N_test)
x_test = np.reshape(x_test, (N_test, 1))
Ktu = compute_kernel(lls, lsf, x_test, z)
Ktt = compute_kernel(lls, lsf, x_test, x_test)
KtuKuuinv = np.dot(Ktu, Kuuinv)
mt = np.dot(KtuKuuinv, mu)
Vt = Ktt - np.dot(KtuKuuinv, Ktu.T) + np.dot(KtuKuuinv, np.dot(Vu, KtuKuuinv.T)) + np.exp(lsn) * np.eye(N_test)
mt = np.reshape(mt, (N_test, ))
vt = np.sqrt(np.diag(Vt).reshape((N_test, )))

def compute_m_v_mm(mx, vx):
    E0 = compute_kernel(lls, lsf, mx, mx)
    E1, E2 = compute_psi_weave(lls, lsf, mx, vx, z)
    mx = np.dot(E1, np.dot(Kuuinv, mu))

    B = np.dot(Kuuinv, np.dot(Vu + np.outer(mu, mu), Kuuinv)) - Kuuinv
    vx = E0 + np.exp(lsn) - mx**2 + np.sum(B * E2)
    vx = vx
    return mx[0, 0], vx[0, 0]

m_ins = [-4, -1, 0, 1.2, 5.5]
v_ins = [0.002, 1.5, 0.04, 0.3, 0.4]


fig = plt.figure(figsize=figsize(1))

gs1 = gridspec.GridSpec(3, 1)
gs1.update(top=0.99, bottom=0.68, hspace=0.0)
ax1 = plt.subplot(gs1[0, :])
ax2 = plt.subplot(gs1[1:, :], sharex=ax1)

gs2 = gridspec.GridSpec(5, 1)
gs2.update(top=0.60, bottom=0.05, hspace=0.0)
ax3 = plt.subplot(gs2[0, :])
ax4 = plt.subplot(gs2[1, :], sharex=ax3)
ax5 = plt.subplot(gs2[2, :], sharex=ax3)
ax6 = plt.subplot(gs2[3, :], sharex=ax3)
ax7 = plt.subplot(gs2[4, :], sharex=ax3)

axs_mm = [ax3, ax4, ax5, ax6, ax7]

ax2.fill_between(x_test.reshape((N_test, )), mt - 2.0*vt, mt + 2.0*vt, color='black', alpha=0.3)
ax2.plot(x_test.reshape((N_test, )), mt, color='black')
ax2.set_ylim([-2.5, 2.5])

ax1.yaxis.set_major_locator(plt.NullLocator())

for i, val in enumerate(m_ins):
    print i

    xplot = x_test.reshape((N_test, ))
    pdfs = mlab.normpdf(xplot, val, np.sqrt(v_ins[i]))
    ax1.plot(xplot, pdfs/np.max(pdfs), color=tableau20[i*2])
    ax1.fill_between(xplot, 0, pdfs/np.max(pdfs), color=tableau20[i*2], alpha=0.4)

    # draw samples
    N_samples = 2000

    youts = np.zeros((N_samples, ))
    m_in = m_ins[i] * np.ones((1, 1))
    v_in = v_ins[i] * np.ones((1, 1))
    for n in range(N_samples):
        x_in = m_in + np.sqrt(v_in) * np.random.randn()
        Ktu = compute_kernel(lls, lsf, x_in, z)
        Ktt = compute_kernel(lls, lsf, x_in, x_in)
        KtuKuuinv = np.dot(Ktu, Kuuinv)
        mtn = np.dot(KtuKuuinv, mu)
        Vtn = Ktt - np.dot(KtuKuuinv, Ktu.T) + np.dot(KtuKuuinv, np.dot(Vu, KtuKuuinv.T)) + np.exp(lsn)

        y_out = mtn + np.sqrt(Vtn[0, 0]) * np.random.randn()

        youts[n] = y_out

    y, binEdges=np.histogram(youts, bins=50, normed=True)

    mypdf = gaussian_kde(youts)
    yplot = np.linspace(-3, 2, 200)
    axs_mm[i].plot(yplot, mypdf(yplot), '-', color='k')

    m_in = m_ins[i] * np.ones((1, 1))
    v_in = v_ins[i] * np.ones((1, 1))
    m_out, v_out = compute_m_v_mm(m_in, v_in)
    N_plot = 300
    xplot = np.linspace(-3, 3, N_plot).reshape((N_plot, ))
    pdfs = mlab.normpdf(xplot, m_out, np.sqrt(v_out))
    axs_mm[i].plot(xplot, pdfs, color=tableau20[i*2], linewidth=2)
    axs_mm[i].fill_between(xplot, 0, pdfs, color=tableau20[i*2], alpha=0.4)
    axs_mm[i].set_xlim([-2.5, 2.5])

    axs_mm[i].yaxis.set_major_locator(plt.NullLocator())
    axs_mm[i].set_ylabel('p(f(x))')

ax7.set_xlabel('f')
ax2.set_xlabel('x')
ax2.set_ylabel('f')
ax1.set_ylabel('p(x)')
ax1.set_xlim([-6, 7])

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax6.get_xticklabels(), visible=False)
    
plt.savefig('/tmp/prop_mm.pdf', bbox_inches='tight')


