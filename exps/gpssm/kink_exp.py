import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../../')
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
import os, sys
import geepee.aep_models as aep
import geepee.vfe_models as vfe
import geepee.ep_models as ep
from geepee.config import PROP_MC, PROP_MM
import pdb

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
    
def plot_latent(model, y, plot_title=''):
    # make prediction on some test inputs
    N_test = 200
    C = model.get_hypers()['C_emission'][0, 0]
    x_test = np.linspace(-4, 6, N_test) / C
    x_test = np.reshape(x_test, [N_test, 1])
    zu = model.dyn_layer.zu
    mu, vu = model.predict_f(zu)
    # mu, Su = model.dyn_layer.mu, model.dyn_layer.Su
    mf, vf = model.predict_f(x_test)
    my, vy = model.predict_y(x_test)
    # plot function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(x_test[:,0], kink_true(x_test[:,0]), '-', color='k')
    ax.plot(C*x_test[:,0], my[:,0], '-', color='r', label='y')
    ax.fill_between(
        C*x_test[:,0], 
        my[:,0] + 2*np.sqrt(vy[:, 0]), 
        my[:,0] - 2*np.sqrt(vy[:, 0]), 
        alpha=0.2, edgecolor='r', facecolor='r')
    # ax.plot(zu, mu, 'ob')
    # ax.errorbar(zu, mu, yerr=3*np.sqrt(vu), fmt='ob')
    # ax.plot(x_test[:,0], mf[:,0], '-', color='b')
    # ax.fill_between(
    #     x_test[:,0], 
    #     mf[:,0] + 2*np.sqrt(vf[:,0]), 
    #     mf[:,0] - 2*np.sqrt(vf[:,0]), 
    #     alpha=0.2, edgecolor='b', facecolor='b')
    ax.plot(
        y[0:model.N-1], 
        y[1:model.N], 
        'r+', alpha=0.5)
    mx, vx = model.get_posterior_x()
    ax.set_xlabel(r'$x_{t-1}$')
    ax.set_ylabel(r'$x_{t}$')
    ax.set_xlim([-4, 6])
    # ax.set_ylim([-7, 7])
    plt.title(plot_title)
    # plt.savefig('/tmp/kink_'+plot_title+'.pdf')
    plt.savefig('/tmp/kink_'+plot_title+'.png')

def plot_prediction_MM(model, y_train, y_test, plot_title=''):
    T = y_test.shape[0]
    mx, vx, my, vy_noiseless, vy = model.predict_forward(T, prop_mode=PROP_MM)
    T_train = y_train.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(T_train), y_train[:, 0], 'k+-')
    ttest = np.arange(T_train, T_train+T)
    # pdb.set_trace()
    ax.plot(ttest, my[:, 0], '-', color='b')
    ax.fill_between(
        ttest, 
        my[:, 0] + 2*np.sqrt(vy_noiseless[:, 0]),
        my[:, 0] - 2*np.sqrt(vy_noiseless[:, 0]),
        alpha=0.3, edgecolor='b', facecolor='b')
    ax.fill_between(
        ttest, 
        my[:, 0] + 2*np.sqrt(vy[:, 0]),
        my[:, 0] - 2*np.sqrt(vy[:, 0]),
        alpha=0.1, edgecolor='b', facecolor='b')
    ax.plot(ttest, y_test, 'ro')
    ax.set_xlim([T_train-5, T_train + T])
    plt.title(plot_title)
    plt.savefig('/tmp/kink_pred_MM_'+plot_title+'.pdf')
    # plt.savefig('/tmp/kink_pred_MM_'+plot_title+'.png')

def find_rank(arr):
    a = np.array(arr)
    r = np.array(a.argsort().argsort(), dtype=float)
    f = a==a
    for i in xrange(len(a)):
        if not f[i]: 
            continue
        s = a == a[i]
        ls = np.sum(s)
        if ls > 1:
            tr = np.sum(r[s])
            r[s] = float(tr)/ls
            f[s] = False
    return r

def compute_log_lik(noise, y_test, prediction):
    loglik = - 0.5 * (prediction - y_test)**2 / noise
    weights = np.arange(y_test.shape[0])**2 + 1
    loglik *= weights
    loglik_sum = np.sum(loglik, axis=1)
    ranks = find_rank(loglik_sum)
    return loglik, (ranks + 1) / np.max(ranks)

def plot_prediction_MC(model, y_train, y_test, plot_title=''):
    T = y_test.shape[0]
    x_samples, my, vy = model.predict_forward(T, prop_mode=PROP_MC)
    T_train = y_train.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(T_train), y_train[:, 0], 'k+-')
    ttest = np.arange(T_train, T_train+T)
    ttest = np.reshape(ttest, [T, 1])
    loglik, ranks = compute_log_lik(np.exp(2*model.sn), y_test, my[:, :, 0].T)
    red = 0.1
    green = 0. * red
    blue = 1. - red
    color = np.array([red, green, blue]).T
    for k in np.argsort(ranks):
        ax.plot(ttest, my[:, k, 0], '-', color=color*ranks[k], alpha=0.5)
    # ax.plot(np.tile(ttest, [1, my.shape[1]]), my[:, :, 0], '-x', color='r', alpha=0.3)
    # ax.plot(np.tile(ttest, [1, my.shape[1]]), x_samples[:, :, 0], 'x', color='m', alpha=0.3)
    ax.plot(ttest, y_test, 'ro')
    ax.set_xlim([T_train-5, T_train + T])
    plt.title(plot_title)
    plt.savefig('/tmp/kink_pred_MC_'+plot_title+'.pdf')
    # plt.savefig('/tmp/kink_pred_MC_'+plot_title+'.png')


# generate a dataset from the kink function above
T = 198
Ttest = 10
process_noise = 0.2
obs_noise = 0.1
(xtrue, x, y) = kink(T+Ttest, process_noise, obs_noise)
y_train = y[:T]
y_test = y[T:]
y_train = np.reshape(y_train, [y_train.shape[0], 1])

Dlatent = 1
Dobs = 1
M = 15

# create VFE model
np.random.seed(42)
model_vfe = vfe.SGPSSM(y_train, Dlatent, M, 
    lik='Gaussian', prior_mean=0, prior_var=1)
vfe_hypers = model_vfe.init_hypers(y_train)
model_vfe.update_hypers(vfe_hypers)
# optimise
# model_vfe.optimise(method='L-BFGS-B', maxiter=10000, reinit_hypers=False)
model_vfe.optimise(method='adam', adam_lr=0.01, maxiter=20000, reinit_hypers=False)
opt_hypers = model_vfe.get_hypers()
plot_latent(model_vfe, y_train, 'VFE')
plot_prediction_MM(model_vfe, y_train, y_test, 'VFE')
plot_prediction_MC(model_vfe, y_train, y_test, 'VFE')

# alphas = [0.001, 0.05, 0.2, 0.5, 1.0]
alphas = [0.001, 0.5, 1]
for alpha in alphas:
    print 'alpha = %.3f' % alpha
    # create AEP model
    np.random.seed(42)
    model_aep = aep.SGPSSM(y_train, Dlatent, M, 
        lik='Gaussian', prior_mean=0, prior_var=1)
    aep_hypers = model_aep.init_hypers(y_train)
    model_aep.update_hypers(aep_hypers)
    # optimise
    # model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=10000, reinit_hypers=False)
    model_aep.optimise(method='adam', alpha=alpha, adam_lr=0.01, maxiter=20000, reinit_hypers=False)
    opt_hypers = model_aep.get_hypers()
    plot_latent(model_aep, y_train, 'AEP_%.3f'%alpha)
    plot_prediction_MM(model_aep, y_train, y_test, 'AEP_%.3f'%alpha)
    plot_prediction_MC(model_aep, y_train, y_test, 'AEP_%.3f'%alpha)

    # create EP model
    model_ep = ep.SGPSSM(y_train, Dlatent, M, 
        lik='Gaussian', prior_mean=0, prior_var=1000)
    # init EP model using the AEP solution
    model_ep.update_hypers(opt_hypers)
    # run EP
    decay = 0.5
    parallel = True
    no_epochs = 2000
    model_ep.inference(no_epochs=no_epochs, alpha=alpha, parallel=parallel, decay=decay)
    plot_latent(model_ep, y_train, 'PEP_%.3f'%alpha)
    plot_prediction_MM(model_ep, y_train, y_test, 'PEP_%.3f'%alpha)
    plot_prediction_MC(model_ep, y_train, y_test, 'PEP_%.3f'%alpha)
    
    # create EP model
    model_ep = ep.SGPSSM(y_train, Dlatent, M, 
        lik='Gaussian', prior_mean=0, prior_var=1000)
    # init EP model using the AEP solution
    model_ep.update_hypers(opt_hypers)
    aep_sgp_layer = model_aep.dyn_layer
    Nm1 = aep_sgp_layer.N
    model_ep.dyn_layer.t1 = 1.0/Nm1 * np.tile(
        aep_sgp_layer.theta_2[np.newaxis, :, :], [Nm1, 1, 1])
    model_ep.dyn_layer.t2 = 1.0/Nm1 * np.tile(
        aep_sgp_layer.theta_1[np.newaxis, :, :, :], [Nm1, 1, 1, 1])
    model_ep.x_prev_1 = np.copy(model_aep.x_factor_1)
    model_ep.x_prev_2 = np.copy(model_aep.x_factor_2)
    model_ep.x_next_1 = np.copy(model_aep.x_factor_1)
    model_ep.x_next_2 = np.copy(model_aep.x_factor_2)
    model_ep.x_up_1 = np.copy(model_aep.x_factor_1)
    model_ep.x_up_2 = np.copy(model_aep.x_factor_2)
    model_ep.x_prev_1[0, :] = 0
    model_ep.x_prev_2[0, :] = 0
    model_ep.x_next_1[-1, :] = 0
    model_ep.x_next_2[-1, :] = 0
    # run EP
    decay = 0.5
    parallel = True
    no_epochs = 2000    
    model_ep.inference(no_epochs=no_epochs, alpha=alpha, parallel=parallel, decay=decay)
    plot_latent(model_ep, y_train, 'PEP_(AEP_init)_%.3f'%alpha)
    plot_prediction_MM(model_ep, y_train, y_test, 'PEP_(AEP_init)_%.3f'%alpha)
    plot_prediction_MC(model_ep, y_train, y_test, 'PEP_(AEP_init)_%.3f'%alpha)