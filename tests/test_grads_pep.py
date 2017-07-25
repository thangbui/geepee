import numpy as np
import copy
import pdb
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from .context import pep, vfe
from .context import pep_tmp as pep
from test_utils import check_grad

np.random.seed(42)

def test_gpr_pep_gaussian():
    N_train = 20
    M = 5
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model = pep.SGPR_rank_one(x_train, y_train, M, lik='Gaussian')
    params = model.init_hypers(y_train)
    print 'gpr pep gaussian'
    for alpha in np.linspace(0.05, 1, 20):
        print alpha
        check_grad(params, model, stochastic=False, alpha=alpha)

def compared_gpr_aep_gaussian_collapsed():
    # make sure that the number of pep sweeps is large in the pep model
    N_train = 20
    M = 10
    D = 1
    Q = 1
    for alpha in np.linspace(0.05, 1, 5):
        y_train = np.random.randn(N_train, Q)
        x_train = np.random.randn(N_train, D)
        model = pep.SGPR_rank_one(x_train, y_train, M, lik='Gaussian')
        params = model.init_hypers(y_train)
        obj, grads = model.objective_function(params, N_train, alpha=alpha)

        collapsed_model = vfe.SGPR_collapsed(x_train, y_train, M)
        c_obj, c_grads = collapsed_model.objective_function(params, N_train, alpha=alpha)

        print alpha
        print obj - c_obj
        print obj, c_obj

        for key in c_grads.keys():
            print key
            print grads[key] - c_grads[key]
            print grads[key]
            print c_grads[key]


def test_gpr_pep_probit():
    N_train = 5
    alpha = 0.5
    M = 3
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = pep.SGPR_rank_one(x_train, y_train, M, lik='Probit')
    params = model.init_hypers(y_train)
    print 'gpr aep probit'
    for alpha in np.linspace(0.05, 1, 20):
        print alpha
        check_grad(params, model, stochastic=False, alpha=alpha)


# TODO
# def plot_gpr_pep_gaussian_stochastic():
#     N_train = 2000
#     alpha = 0.5
#     M = 50
#     D = 2
#     Q = 3
#     y_train = np.random.randn(N_train, Q)
#     x_train = np.random.randn(N_train, D)
#     model = aep.SGPR(x_train, y_train, M, lik='Gaussian')

#     # init hypers, inducing points and q(u) params
#     params = model.init_hypers(y_train)
#     logZ, grad_all = model.objective_function(params, N_train, alpha=alpha)
#     mbs = np.logspace(-2, 0, 10)
#     reps = 20
#     times = np.zeros(len(mbs))
#     objs = np.zeros((len(mbs), reps))
#     for i, mb in enumerate(mbs):
#         no_points = int(N_train * mb)
#         start_time = time.time()
#         for k in range(reps):
#             objs[i, k] = model.objective_function(
#                 params, no_points, alpha=alpha)[0]
#         times[i] = time.time() - start_time

#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#     ax1.plot(mbs, times, 'x-')
#     ax1.set_xlabel("Minibatch proportion")
#     ax1.set_ylabel("Time taken")
#     ax1.set_xscale("log", nonposx='clip')

#     ax2.plot(mbs, objs, 'kx')
#     ax2.axhline(logZ, color='b')
#     ax2.set_xlabel("Minibatch proportion")
#     ax2.set_ylabel("ELBO estimates")
#     ax2.set_xscale("log", nonposx='clip')
#     plt.savefig('/tmp/gaussian_stochastic_aep_gpr.pdf')


# def plot_gpr_pep_probit_stochastic():
#     N_train = 2000
#     alpha = 0.5
#     M = 50
#     D = 2
#     Q = 3
#     x_train = np.random.randn(N_train, D)
#     y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
#     model = aep.SGPR(x_train, y_train, M, lik='Probit')

#     # init hypers, inducing points and q(u) params
#     params = model.init_hypers(y_train)
#     logZ, grad_all = model.objective_function(params, N_train, alpha=alpha)
#     mbs = np.logspace(-2, 0, 10)
#     reps = 20
#     times = np.zeros(len(mbs))
#     objs = np.zeros((len(mbs), reps))
#     for i, mb in enumerate(mbs):
#         no_points = int(N_train * mb)
#         start_time = time.time()
#         for k in range(reps):
#             objs[i, k] = model.objective_function(
#                 params, no_points, alpha=alpha)[0]
#         times[i] = time.time() - start_time

#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#     ax1.plot(mbs, times, 'x-')
#     ax1.set_xlabel("Minibatch proportion")
#     ax1.set_ylabel("Time taken")
#     ax1.set_xscale("log", nonposx='clip')

#     ax2.plot(mbs, objs, 'kx')
#     ax2.axhline(logZ, color='b')
#     ax2.set_xlabel("Minibatch proportion")
#     ax2.set_ylabel("ELBO estimates")
#     ax2.set_xscale("log", nonposx='clip')
#     plt.savefig('/tmp/probit_stochastic_aep_gpr.pdf')


if __name__ == '__main__':

    # compared_gpr_aep_gaussian_collapsed()

    for i in range(1):
        test_gpr_pep_gaussian()
        test_gpr_pep_probit()

    # plot_gpr_pep_probit_stochastic()
    # plot_gpr_pep_gaussian_stochastic()