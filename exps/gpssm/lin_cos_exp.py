import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
import os, sys
from ..context import aep, ep
import pdb

np.random.seed(42)

def func_true(x):
    fx = -0.5*x + 5*np.cos(0.5*x)
    return fx

def func(T, process_noise, obs_noise, xprev=None):
    if xprev is None:
        xprev = np.random.randn()
    y = np.zeros([T, ])
    x = np.zeros([T, ])
    xtrue = np.zeros([T, ])
    for t in range(T):
        fx = -0.5*xprev + 5*np.cos(0.5*xprev)
        xtrue[t] = fx
        x[t] = fx + np.sqrt(process_noise)*np.random.randn()
        xprev = x[t]
        y[t] = 0.5*x[t] + np.sqrt(obs_noise)*np.random.randn()

    return xtrue, x, y

T = 200
process_noise = 0.3
obs_noise = 0.2
(xtrue, x, y) = func(T, process_noise, obs_noise)
y_train = np.reshape(y, [y.shape[0], 1])

alpha = 0.5
Dlatent = 1
Dobs = 1
M = 30
C = 0.5*np.ones((1, 1))
R = np.ones(1)*np.log(obs_noise)/2
lls = np.reshape(np.log(2), [Dlatent, ])
lsf = np.reshape(np.log(2), [1, ])
zu = np.linspace(-8, 6, M)
zu = np.reshape(zu, [M, 1])
lsn = np.log(0.01)/2
params = {'ls': lls, 'sf': lsf, 'sn': lsn, 'R': R, 'C': C, 'zu': zu}

# create model
model = aep.SGPSSM(y_train, Dlatent, M, 
    lik='Gaussian', prior_mean=0, prior_var=1000)
hypers = model.init_hypers(y_train)
for key in params.keys():
    hypers[key] = params[key]
model.update_hypers(hypers, alpha)
model.set_fixed_params(['C', 'R'])
model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=10000, reinit_hypers=False)

def plot_latent(model, latent_true, fname):
    # make prediction on some test inputs
    N_test = 500
    x_test = np.linspace(-10, 8, N_test)
    x_test = np.reshape(x_test, [N_test, 1])
    zu = model.sgp_layer.zu
    mu, vu = model.predict_f(zu)
    mf, vf = model.predict_f(x_test)
    # plot function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_test[:,0], func_true(x_test[:,0]), '-', color='k')
    ax.plot(zu, mu, 'ob')
    ax.plot(x_test[:,0], mf[:,0], '-', color='b', label='f, alpha=%.2f' % alpha)
    ax.fill_between(
        x_test[:,0], 
        mf[:,0] + 2*np.sqrt(vf[:,0]), 
        mf[:,0] - 2*np.sqrt(vf[:,0]), 
        alpha=0.2, edgecolor='b', facecolor='b')
    ax.plot(
        latent_true[0:model.N-1], 
        latent_true[1:model.N], 
        'r+', alpha=0.5)
    mx, vx = model.get_posterior_x()
    ax.plot(mx[0:model.N-1], mx[1:model.N], 'og', alpha=0.3)
    ax.set_xlabel(r'$x_{t-1}$')
    ax.set_ylabel(r'$x_{t}$')
    ax.set_xlim([-10, 8])
    ax.legend(loc='lower center')
    plt.savefig('/tmp/'+fname+'gpssm_lin_cos_%.2f.pdf'%alpha)
    
    # plot function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mx, vx = model.get_posterior_x()
    ax.plot(np.arange(model.N), mx, '-g', alpha=0.5)
    ax.fill_between(
        np.arange(model.N), 
        mx[:,0] + 2*np.sqrt(vx[:,0]), 
        mx[:,0] - 2*np.sqrt(vx[:,0]), 
        alpha=0.3, edgecolor='g', facecolor='g')
    ax.plot(np.arange(model.N), latent_true, 'r+', alpha=0.5)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x_{t}$')
    ax.set_xlim([0, model.N])
    ax.legend(loc='lower center')
    
    se = (latent_true - mx[:, 0])**2
    mse = np.mean(se)
    se_std = np.std(se)/np.sqrt(se.shape[0])
    
    ll = -0.5 * (latent_true - mx[:, 0])**2/vx[:, 0] -0.5*np.log(2*np.pi*vx[:, 0])
    mll = np.mean(ll)
    ll_std = np.std(ll)/np.sqrt(ll.shape[0])
    print 'se %.3f +/- %.3f' % (mse, se_std)
    print 'll %.3f +/- %.3f' % (mll, ll_std)

plot_latent(model, xtrue, 'aep')


# create EP model
opt_hypers = model.get_hypers()
model_ep = ep.SGPSSM(y_train, Dlatent, M, 
    lik='Gaussian', prior_mean=0, prior_var=1000)
model_ep.update_hypers(opt_hypers)
# run EP
model_ep.inference(no_epochs=100, alpha=alpha, parallel=True, decay=0.99)
plot_latent(model_ep, xtrue, 'ep')

