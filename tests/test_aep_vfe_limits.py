import numpy as np
from scipy.optimize import check_grad
import copy
import pdb
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from .context import vfe, aep
from .context import flatten_dict, unflatten_dict
from .context import PROP_MC, PROP_MM

np.random.seed(0)

def test_gpr_gaussian(nat_param=True):
    N_train = 20
    alpha = 0.000001
    M = 10
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model_vfe = vfe.SGPR(x_train, y_train, M, lik='Gaussian', nat_param=nat_param)
    params = model_vfe.init_hypers(y_train)
    params = model_vfe.optimise(method='adam', maxiter=2000)
    logZ_vfe, _ = model_vfe.objective_function(params, N_train)

    model_aep = aep.SGPR(x_train, y_train, M, lik='Gaussian', nat_param=nat_param)
    logZ_aep, _ = model_aep.objective_function(params, N_train, alpha=alpha)

    d = np.abs(logZ_aep - logZ_vfe)
    print 'gpr gaussian aep %.4f, vfe %.4f, diff %.4f' % (logZ_aep, logZ_vfe, d)   


def test_gpr_probit(nat_param=True):
    N_train = 5
    alpha = 0.000001
    M = 3
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model_vfe = vfe.SGPR(x_train, y_train, M, lik='Probit', nat_param=nat_param)
    params = model_vfe.init_hypers(y_train)
    logZ_vfe, _ = model_vfe.objective_function(params, N_train)

    model_aep = aep.SGPR(x_train, y_train, M, lik='Probit', nat_param=nat_param)
    logZ_aep, _ = model_aep.objective_function(params, N_train, alpha=alpha)

    d = np.abs(logZ_aep - logZ_vfe)
    print 'gpr probit aep %.4f, vfe %.4f, diff %.4f' % (logZ_aep, logZ_vfe, d)


def test_gplvm_gaussian(nat_param=True):
    N_train = 5
    alpha = 0.000001
    M = 4
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    model_vfe = vfe.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)
    params = model_vfe.init_hypers(y_train)
    params = model_vfe.optimise(method='adam', maxiter=500)
    logZ_vfe, _ = model_vfe.objective_function(params, N_train)

    model_aep = aep.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)
    logZ_aep, _ = model_aep.objective_function(params, N_train, alpha=alpha)

    d = np.abs(logZ_aep - logZ_vfe)
    print 'gplvm gaussian MM aep %.4f, vfe %.4f, diff %.4f' % (logZ_aep, logZ_vfe, d)


def test_gplvm_probit(nat_param=True):
    N_train = 5
    alpha = 0.000001
    M = 3
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model_vfe = vfe.SGPLVM(y_train, D, M, lik='Probit', nat_param=nat_param)
    params = model_vfe.init_hypers(y_train)
    logZ_vfe, _ = model_vfe.objective_function(params, N_train)

    model_aep = aep.SGPLVM(y_train, D, M, lik='Probit', nat_param=nat_param)
    logZ_aep, _ = model_aep.objective_function(params, N_train, alpha=alpha)

    d = np.abs(logZ_aep - logZ_vfe)
    print 'gplvm probit MM aep %.4f, vfe %.4f, diff %.4f' % (logZ_aep, logZ_vfe, d)


def test_gplvm_gaussian_MC(nat_param=True):
    N_train = 5
    alpha = 0.000001
    M = 4
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    model_vfe = vfe.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)
    params = model_vfe.init_hypers(y_train)
    params = model_vfe.optimise(method='adam', maxiter=50)
    logZ_vfe, _ = model_vfe.objective_function(params, N_train, prop_mode=PROP_MC)

    model_aep = aep.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)
    logZ_aep, _ = model_aep.objective_function(params, N_train, alpha=alpha, prop_mode=PROP_MC)

    d = np.abs(logZ_aep - logZ_vfe)
    print 'gplvm gaussian MC aep %.4f, vfe %.4f, diff %.4f' % (logZ_aep, logZ_vfe, d)


def test_gplvm_probit_MC(nat_param=True):
    N_train = 5
    alpha = 0.000001
    M = 3
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model_vfe = vfe.SGPLVM(y_train, D, M, lik='Probit', nat_param=nat_param)
    params = model_vfe.init_hypers(y_train)
    logZ_vfe, _ = model_vfe.objective_function(params, N_train, prop_mode=PROP_MC)

    model_aep = aep.SGPLVM(y_train, D, M, lik='Probit', nat_param=nat_param)
    logZ_aep, _ = model_aep.objective_function(params, N_train, alpha=alpha, prop_mode=PROP_MC)

    d = np.abs(logZ_aep - logZ_vfe)
    print 'gplvm probit MC aep %.4f, vfe %.4f, diff %.4f' % (logZ_aep, logZ_vfe, d)
    

def test_gpssm_linear_gaussian_kink_MM(nat_param=True):
    np.random.seed(0)
    def kink_true(x):
        fx = np.zeros(x.shape)
        for t in range(x.shape[0]):
            xt = x[t]
            if xt < 4:
                fx[t] = xt + 1
            else:
                fx[t] = -4 * xt + 21
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
                fx = -4 * xprev + 21

            xtrue[t] = fx
            x[t] = fx + np.sqrt(process_noise) * np.random.randn()
            xprev = x[t]
            y[t] = x[t] + np.sqrt(obs_noise) * np.random.randn()
        return xtrue, x, y

    N_train = 100
    process_noise = 0.2
    obs_noise = 0.1
    alpha = 0.00000001
    M = 4
    Q = 1
    D = 1
    (xtrue, x, y) = kink(N_train, process_noise, obs_noise)
    y_train = np.reshape(y, [y.shape[0], 1])
    model_vfe = vfe.SGPSSM(y_train, Q, M, lik='Gaussian', gp_emi=False, nat_param=nat_param)
    
    # init hypers, inducing points and q(u) params
    # params = model_vfe.init_hypers(y_train)
    params = model_vfe.optimise(method='adam', maxiter=500, adam_lr=0.08, disp=False)
    logZ_vfe, _ = model_vfe.objective_function(params, N_train)

    model_aep = aep.SGPSSM(y_train, Q, M, lik='Gaussian', gp_emi=False)
    logZ_aep, _ = model_aep.objective_function(params, N_train, alpha=alpha)

    d = np.abs(logZ_aep - logZ_vfe)
    print 'gpssm gaussian MM aep %.4f, vfe %.4f, diff %.4f' % (logZ_aep, logZ_vfe, d)

if __name__ == '__main__':
    # test_gpr_gaussian(True)
    # # test_gpr_gaussian(False)
    
    # test_gpr_probit(True)
    # # test_gpr_probit(False)
    
    # test_gplvm_gaussian(True)
    # # test_gplvm_gaussian(False)
    
    # test_gplvm_probit(True)
    # # test_gplvm_probit(False)

    # test_gplvm_gaussian_MC(True)
    # test_gplvm_gaussian_MC(False)
    
    # test_gplvm_probit_MC(True)
    # test_gplvm_probit_MC(False)
    
    # test_gpssm_linear_gaussian_kink_MM(True)
    # test_gpssm_linear_gaussian_kink_MM(False)