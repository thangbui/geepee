import numpy as np
import scipy.linalg as spla
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.use('pgf')
from sklearn.neighbors import KernelDensity
from scipy.stats.kde import gaussian_kde

from geepee.kernels import *

import pdb


np.random.seed(100)

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
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
# plt.style.use('ggplot')

# I make my own newfig and savefig functions
def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))

    gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,4],
                       height_ratios=[4,1]
                       )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[3])

    return fig, (ax1, ax2, ax3)

def savefig(filename):
    # plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))


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
lsn = np.log(0.05)

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
N_test = 200
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


def compute_m_v_mm_approx(mx, vx):
    E0 = compute_kernel(lls, lsf, mx, mx)
    E1, E2 = compute_psi_weave(lls, lsf, mx, vx, z)
    E2 = np.diag(np.diag(E2[0, :, :]) - E1**2) + np.outer(E1, E1)
    mx = np.dot(E1, np.dot(Kuuinv, mu))

    B = np.dot(Kuuinv, np.dot(Vu + np.outer(mu, mu), Kuuinv)) - Kuuinv
    vx = E0 + np.exp(lsn) - mx**2 + np.sum(B * E2)
    vx = vx
    return mx[0, 0], vx[0, 0]


def compute_m_v_lin(mx, vx):
    ksu = compute_kernel(lls, lsf, mx, z)
    kss = np.exp(lls) * np.ones((mx.shape[0], 1))

    ms = np.dot(ksu, np.dot(Kuuinv, mu))
    Kuuinv_kus = np.dot(Kuuinv, ksu.T)

    vs = kss - np.sum(ksu * Kuuinv_kus.T, axis=1, keepdims=True)
    vs = vs + np.sum(Kuuinv_kus.T * np.dot(Vu, Kuuinv_kus), axis=1, keepdims=True)

    dK = grad_x(lls, lsf, mx, z)
    g = np.einsum('nmd,ma->nd', dK, np.dot(Kuuinv, mu))
    
    m = ms
    v = g*vx*g + vs

    return m[0, 0], v[0, 0]

m_ins = [-4, -1, 0, 1.2, 5.5]
v_ins = [0.2, 1.5, 0.04, 0.3, 0.4]

for i, val in enumerate(m_ins):
    print i
    fig, ax = newfig(1)

    ax[0].xaxis.set_major_locator(plt.NullLocator())
    ax[2].yaxis.set_major_locator(plt.NullLocator())

    ax[0].set_ylabel('y', rotation=90)
    ax[2].set_xlabel('x')

    ax[1].fill_between(x_test.reshape((N_test, )), mt - 2.0*vt, mt + 2.0*vt, color='black', alpha=0.3)
    ax[1].plot(x_test.reshape((N_test, )), mt, color='black')
    ax[1].set_ylim([-3, 3])
    ax[1].set_xlim([-6, 7])
    ax[2].set_xlim([-6, 7])
    ax[0].set_ylim([-3, 3])

    xplot = x_test.reshape((N_test, ))
    pdfs = mlab.normpdf(xplot, val, v_ins[i])
    ax[2].plot(xplot, pdfs, color=tableau20[i*2])
    ax[2].fill_between(xplot, 0, pdfs, color=tableau20[i*2], alpha=0.4)
    ax[2].set_xlim([-6, 7])

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

        ax[1].plot(x_in, y_out, 'o', color=tableau20[i*2], alpha=0.1)

    y, binEdges=np.histogram(youts, bins=50, normed=True)

    mypdf = gaussian_kde(youts)
    yplot = np.linspace(-3, 2, 200)
    ax[0].plot(mypdf(yplot), yplot, '-', color='k')
    ax[0].set_ylim([-3, 3])

    m_in = m_ins[i] * np.ones((1, 1))
    v_in = v_ins[i] * np.ones((1, 1))
    m_out_app, v_out_app = compute_m_v_mm_approx(m_in, v_in)
    m_out, v_out = compute_m_v_mm(m_in, v_in)
    m_out_lin, v_out_lin = compute_m_v_lin(m_in, v_in)
    
    N_plot = 100
    xplot = np.linspace(-3, 3, N_plot).reshape((N_plot, ))
    pdfs = mlab.normpdf(xplot, m_out, v_out)
    pdfs_app = mlab.normpdf(xplot, m_out_app, v_out_app)
    pdfs_lin = mlab.normpdf(xplot, m_out_lin, v_out_lin)
    ax[0].plot(pdfs, xplot, color=tableau20[i*2], linewidth=2)
    ax[0].plot(pdfs_app, xplot, '--', color=tableau20[i*2], linewidth=2)
    ax[0].plot(pdfs_lin, xplot, '-.', color=tableau20[i*2], linewidth=2)
    ax[0].fill_betweenx(xplot, pdfs, 0, color=tableau20[i*2], alpha=0.4)
    ax[0].set_ylim([-3, 3])
    
    savefig('./tmp/prop_mm_' + str(i))



