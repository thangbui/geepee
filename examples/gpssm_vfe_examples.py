import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
import os
import sys
from .context import vfe
from .context import config
import pdb

np.random.seed(42)

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


def plot_latent_kink(model, y, plot_title=''):
    # make prediction on some test inputs
    N_test = 200
    x_test = np.linspace(-4, 6, N_test) / model.emi_layer.C[0, 0]
    x_test = np.reshape(x_test, [N_test, 1])
    zu = model.dyn_layer.zu
    mu, vu = model.predict_f(zu)
    mf, vf = model.predict_f(x_test)
    my, vy = model.predict_y(x_test)
    # plot function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_test[:,0], kink_true(x_test[:,0]), '-', color='k')
    ax.plot(zu, mu, 'ob')
    ax.plot(x_test[:,0], mf[:,0], '-', color='b')
    ax.fill_between(
        x_test[:,0], 
        mf[:,0] + 2*np.sqrt(vf[:,0]), 
        mf[:,0] - 2*np.sqrt(vf[:,0]), 
        alpha=0.2, edgecolor='b', facecolor='b')

    ax.plot(model.emi_layer.C[0, 0]*x_test[:,0], my[:,0], '-', color='r')
    ax.fill_between(
        model.emi_layer.C[0, 0]*x_test[:,0], 
        my[:,0] + 2*np.sqrt(vy[:,0]), 
        my[:,0] - 2*np.sqrt(vy[:,0]), 
        alpha=0.2, edgecolor='r', facecolor='r')
    ax.plot(
        y[0:model.N-1], 
        y[1:model.N], 
        'r+', alpha=0.5)

    # mx, vx = model.get_posterior_x()
    # ax.plot(mx[0:model.N-1], mx[1:model.N], 'og', alpha=0.3)
    my, vy_noiseless, vy = model.get_posterior_y()
    ax.plot(my[0:model.N-1], my[1:model.N], 'or', alpha=0.2)
    
    ax.set_xlabel(r'$x_{t-1}$')
    ax.set_ylabel(r'$x_{t}$')
    ax.set_xlim([-4, 6])
    ax.set_ylim([-7, 7])
    plt.title(plot_title)
    plt.savefig('/tmp/kink_'+plot_title+'.pdf')


def test_kink_linear_MM():
    # generate a dataset from the kink function above
    T = 200
    process_noise = 0.2
    obs_noise = 0.1
    (xtrue, x, y) = kink(T, process_noise, obs_noise)
    y_train = np.reshape(y, [y.shape[0], 1])

    # init hypers
    Dlatent = 1
    Dobs = 1
    M = 15
    # create vfe model
    model_vfe = vfe.SGPSSM(y_train, Dlatent, M, 
        lik='Gaussian', prior_mean=0, prior_var=1000, gp_emi=False, nat_param=True)
    hypers = model_vfe.init_hypers(y_train)
    model_vfe.update_hypers(hypers)
    # optimise
    # model_vfe.optimise(
    #     method='L-BFGS-B', maxiter=10000, reinit_hypers=False)
    model_vfe.optimise(
        method='adam', maxiter=10000, adam_lr=0.05, reinit_hypers=False)
    opt_hypers = model_vfe.get_hypers()
    plot_latent_kink(model_vfe, y, 'VFE_MM')


def test_kink_linear_MC():
    # generate a dataset from the kink function above
    T = 200
    process_noise = 0.2
    obs_noise = 0.1
    (xtrue, x, y) = kink(T, process_noise, obs_noise)
    y_train = np.reshape(y, [y.shape[0], 1])

    # init hypers
    Dlatent = 1
    Dobs = 1
    M = 15
    # create VFE model
    model_vfe = vfe.SGPSSM(y_train, Dlatent, M, 
        lik='Gaussian', prior_mean=0, prior_var=1000, gp_emi=False)
    hypers = model_vfe.init_hypers(y_train)
    model_vfe.update_hypers(hypers)
    # optimise
    # model_vfe.optimise(
    #     method='L-BFGS-B', maxiter=10000, 
    #     reinit_hypers=False, prop_mode=config.PROP_MC)
    model_vfe.optimise(
        method='adam', maxiter=10000, adam_lr=0.05, 
        reinit_hypers=False, prop_mode=config.PROP_MC)
    opt_hypers = model_vfe.get_hypers()
    plot_latent_kink(model_vfe, y, 'VFE_MC')

if __name__ == '__main__':
    np.random.seed(42)
    test_kink_linear_MM()
    # np.random.seed(42)
    # test_kink_linear_MC()
