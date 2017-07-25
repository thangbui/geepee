import numpy as np
from scipy.optimize import check_grad
import copy
import pdb
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from .context import vfe
from .context import flatten_dict, unflatten_dict
from .context import PROP_MC, PROP_MM
from test_utils import kink, check_grad

np.random.seed(42)
        
def test_gplvm_vfe_gaussian(nat_param=True, stoc=False, prop_mode=PROP_MM):
    N_train = 10
    M = 5
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)
    params = lvm.init_hypers(y_train)
    print 'gplvm vfe gaussian nat_param %r stoc %r prop_mode %s' % (nat_param, stoc, prop_mode)
    check_grad(params, lvm, stochastic=stoc, prop_mode=prop_mode)


def test_gplvm_vfe_probit(nat_param=False, stoc=False, prop_mode=PROP_MM):
    N_train = 5
    M = 3
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    lvm = vfe.SGPLVM(y_train, D, M, lik='Probit', nat_param=nat_param)
    params = lvm.init_hypers(y_train)
    print 'gplvm vfe probit nat_param %r stoc %r prop_mode %s' % (nat_param, stoc, prop_mode)
    check_grad(params, lvm, stochastic=stoc, prop_mode=prop_mode)


def plot_gplvm_vfe_gaussian_stochastic():
    N_train = 2000
    M = 50
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    model = vfe.SGPLVM(y_train, D, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    params = model.init_hypers(y_train)
    logZ, grad_all = model.objective_function(params, N_train)
    mbs = np.logspace(-2, 0, 10)
    reps = 20
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            objs[i, k] = model.objective_function(
                params, no_points, alpha=alpha)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/gaussian_stochastic_vfe_gplvm.pdf')


def plot_gplvm_vfe_probit_stochastic():
    N_train = 2000
    M = 50
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = vfe.SGPLVM(y_train, D, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    params = model.init_hypers(y_train)
    logZ, grad_all = model.objective_function(params, N_train)
    mbs = np.logspace(-2, 0, 10)
    reps = 20
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            objs[i, k] = model.objective_function(
                params, no_points, alpha=alpha)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/probit_stochastic_vfe_gplvm.pdf')


def test_gpr_vfe_gaussian_collapsed():
    N_train = 20
    alpha = 0.3
    M = 10
    D = 2
    Q = 1
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model = vfe.SGPR_collapsed(x_train, y_train, M)
    params = model.init_hypers(y_train)
    print 'gpr vfe gaussian collapsed'
    check_grad(params, model, alpha=alpha)


def test_gpr_vfe_gaussian(nat_param=True, stoc=False):
    N_train = 20
    M = 10
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model = vfe.SGPR(x_train, y_train, M, lik='Gaussian', nat_param=nat_param)
    params = model.init_hypers(y_train)
    print 'gpr vfe gaussian nat_param %r stoc %r' % (nat_param, stoc)
    check_grad(params, model, stochastic=stoc)


def test_gpr_vfe_probit(nat_param=True, stoc=False):
    N_train = 5
    M = 3
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = vfe.SGPR(x_train, y_train, M, lik='Probit', nat_param=nat_param)
    params = model.init_hypers(y_train)
    print 'gpr vfe probit nat_param %r stoc %r' % (nat_param, stoc)
    check_grad(params, model, stochastic=stoc)


def plot_gpr_vfe_gaussian_stochastic():
    N_train = 2000
    M = 50
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model = vfe.SGPR(x_train, y_train, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    params = model.init_hypers(y_train)
    logZ, grad_all = model.objective_function(params, N_train)
    mbs = np.logspace(-2, 0, 10)
    reps = 20
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            objs[i, k] = model.objective_function(
                params, no_points, alpha=alpha)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/gaussian_stochastic_vfe_gpr.pdf')


def plot_gpr_vfe_probit_stochastic():
    N_train = 2000
    alpha = 0.5
    M = 50
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = vfe.SGPR(x_train, y_train, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    params = model.init_hypers(y_train)
    logZ, grad_all = model.objective_function(params, N_train)
    mbs = np.logspace(-2, 0, 10)
    reps = 20
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            objs[i, k] = model.objective_function(
                params, no_points, alpha=alpha)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/probit_stochastic_vfe_gpr.pdf')


def test_dgpr_vfe_gaussian(nat_param=True, stoc=False):
    N_train = 10
    alpha = 1
    M = 5
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    hidden_size = [3, 2]
    # TODO: nat_param
    model = vfe.SDGPR(x_train, y_train, M, hidden_size, lik='Gaussian')
    params = model.init_hypers(y_train)
    print 'dgpr vfe gaussian nat_param %r stoc %r' % (nat_param, stoc)
    check_grad(params, model, stochastic=stoc, alpha=alpha)


def test_dgpr_vfe_probit(nat_param=True, stoc=False):
    N_train = 5
    alpha = 1
    M = 3
    D = 2
    Q = 3
    hidden_size = [3, 2]
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    # TODO: nat_param
    model = vfe.SDGPR(x_train, y_train, M, hidden_size, lik='Probit')
    params = model.init_hypers(y_train)
    print 'dgpr vfe probit nat_param %r stoc %r' % (nat_param, stoc)
    check_grad(params, model, stochastic=stoc, alpha=alpha)


def plot_dgpr_vfe_gaussian_stochastic():
    N_train = 2000
    M = 50
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    hidden_size = [3, 2]
    model = vfe.SDGPR(x_train, y_train, M, hidden_size, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    params = model.init_hypers(y_train)
    logZ, grad_all = model.objective_function(params, N_train)
    mbs = np.logspace(-2, 0, 10)
    reps = 20
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            objs[i, k] = model.objective_function(
                params, no_points, alpha=1.0)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/gaussian_stochastic_vfe_dgpr.pdf')


def plot_dgpr_vfe_probit_stochastic():
    N_train = 2000
    M = 50
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    hidden_size = [3, 2]
    model = vfe.SDGPR(x_train, y_train, M, hidden_size, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    params = model.init_hypers(y_train)
    logZ, grad_all = model.objective_function(params, N_train)
    mbs = np.logspace(-2, 0, 10)
    reps = 20
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            objs[i, k] = model.objective_function(
                params, no_points, alpha=1.0)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/probit_stochastic_vfe_dgpr.pdf')


def test_dgprh_vfe_gaussian(nat_param=True, stoc=False):
    N_train = 10
    M = 5
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    hidden_size = [3, 2]
    model = vfe.SDGPR_H(x_train, y_train, M, hidden_size, lik='Gaussian')
    params = model.init_hypers(y_train)
    print 'dgprh vfe gaussian nat_param %r stoc %r' % (nat_param, stoc)
    check_grad(params, model, stochastic=stoc)


def test_dgprh_vfe_probit(nat_param=True, stoc=False):
    N_train = 5
    M = 3
    D = 2
    Q = 3
    hidden_size = [3, 2]
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = vfe.SDGPR_H(x_train, y_train, M, hidden_size, lik='Probit')
    params = model.init_hypers(y_train)
    print 'dgprh vfe probit nat_param %r stoc %r' % (nat_param, stoc)
    check_grad(params, model, stochastic=stoc)


def test_gpssm_linear_vfe_gaussian_kink(nat_param=True, stoc=False, prop_mode=PROP_MM):
    N_train = 50
    process_noise = 0.2
    obs_noise = 0.1
    M = 4
    Q = 1
    D = 1
    (xtrue, x, y) = kink(N_train, process_noise, obs_noise)
    y_train = np.reshape(y, [y.shape[0], 1])
    # TODO nat_param=nat_param
    lvm = vfe.SGPSSM(y_train, Q, M, lik='Gaussian', gp_emi=False, nat_param=nat_param)

    lvm.optimise(method='adam', maxiter=1000, adam_lr=0.1)
    params = lvm.get_hypers()

    # # init hypers, inducing points and q(u) params
    # params = lvm.init_hypers(y_train)
    
    print 'gplvm linear emis vfe kink nat_param %r stoc %r prop_mode %s' % (nat_param, stoc, prop_mode)
    check_grad(params, lvm, stochastic=stoc, prop_mode=prop_mode)


def test_gpssm_gp_vfe_gaussian_kink(nat_param=True, stoc=False, prop_mode=PROP_MM):
    N_train = 10
    process_noise = 0.2
    obs_noise = 0.1
    M = 4
    Q = 1
    D = 1
    (xtrue, x, y) = kink(N_train, process_noise, obs_noise)
    y_train = np.reshape(y, [y.shape[0], 1])
    lvm = vfe.SGPSSM(y_train, Q, M, lik='Gaussian', gp_emi=True, nat_param=nat_param)

    lvm.optimise(method='adam', maxiter=1000, adam_lr=0.1)
    params = lvm.get_hypers()

    # # init hypers, inducing points and q(u) params
    # params = lvm.init_hypers(y_train)
    
    print 'gplvm gp emis vfe kink nat_param %r stoc %r prop_mode %s' % (nat_param, stoc, prop_mode)
    check_grad(params, lvm, stochastic=stoc, prop_mode=prop_mode)


def plot_gpssm_linear_vfe_gaussian_stochastic():
    N_train = 2000
    M = 50
    Q = 2
    D = 3
    y_train = np.random.randn(N_train, D)
    model = vfe.SGPSSM(y_train, Q, M, lik='Gaussian', gp_emi=False)

    # init hypers, inducing points and q(u) params
    params = model.init_hypers(y_train)
    logZ, grad_all = model.objective_function(params, N_train)
    mbs = np.logspace(-2, 0, 20)
    reps = 40
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            print '%d/%d, %d/%d' % (i, len(mbs), k, reps)
            objs[i, k] = model.objective_function(
                params, no_points, alpha=alpha, prop_mode=PROP_MM)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/gaussian_stochastic_vfe_gpssm_linear_MM.pdf')

    # init hypers, inducing points and q(u) params
    logZ, grad_all = model.objective_function(params, N_train)
    times = np.zeros(len(mbs))
    objs = np.zeros((len(mbs), reps))
    for i, mb in enumerate(mbs):
        no_points = int(N_train * mb)
        start_time = time.time()
        for k in range(reps):
            print '%d/%d, %d/%d' % (i, len(mbs), k, reps)
            objs[i, k] = model.objective_function(
                params, no_points, alpha=alpha, prop_mode=PROP_MC)[0]
        times[i] = time.time() - start_time

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(mbs, times, 'x-')
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")
    ax1.set_xscale("log", nonposx='clip')

    ax2.plot(mbs, objs, 'kx')
    ax2.axhline(logZ, color='b')
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    ax2.set_xscale("log", nonposx='clip')
    plt.savefig('/tmp/gaussian_stochastic_vfe_gpssm_linear_MC.pdf')


if __name__ == '__main__':
    # TODO: MC prop mode
    # test_gplvm_vfe_gaussian(nat_param=True, stoc=False, prop_mode=PROP_MM)
    # test_gplvm_vfe_gaussian(nat_param=True, stoc=True, prop_mode=PROP_MM)
    # test_gplvm_vfe_gaussian(nat_param=False, stoc=False, prop_mode=PROP_MM)
    # test_gplvm_vfe_gaussian(nat_param=False, stoc=True, prop_mode=PROP_MM)

    # test_gplvm_vfe_gaussian(nat_param=True, stoc=False, prop_mode=PROP_MC)
    # test_gplvm_vfe_gaussian(nat_param=True, stoc=True, prop_mode=PROP_MC)
    # test_gplvm_vfe_gaussian(nat_param=False, stoc=False, prop_mode=PROP_MC)
    # test_gplvm_vfe_gaussian(nat_param=False, stoc=True, prop_mode=PROP_MC)

    # test_gplvm_vfe_probit(nat_param=True, stoc=False, prop_mode=PROP_MM)
    # test_gplvm_vfe_probit(nat_param=True, stoc=True, prop_mode=PROP_MM)
    # test_gplvm_vfe_probit(nat_param=False, stoc=False, prop_mode=PROP_MM)
    # test_gplvm_vfe_probit(nat_param=False, stoc=True, prop_mode=PROP_MM)
    
    # test_gplvm_vfe_probit(nat_param=True, stoc=False, prop_mode=PROP_MC)
    # test_gplvm_vfe_probit(nat_param=True, stoc=True, prop_mode=PROP_MC)
    # test_gplvm_vfe_probit(nat_param=False, stoc=False, prop_mode=PROP_MC)
    # test_gplvm_vfe_probit(nat_param=False, stoc=True, prop_mode=PROP_MC)

    # plot_gplvm_vfe_probit_stochastic()
    # plot_gplvm_vfe_gaussian_stochastic()

    # test_gpr_vfe_gaussian_collapsed()

    # test_gpr_vfe_gaussian(nat_param=True, stoc=False)
    # test_gpr_vfe_gaussian(nat_param=True, stoc=True)
    # test_gpr_vfe_gaussian(nat_param=False, stoc=False)
    # test_gpr_vfe_gaussian(nat_param=False, stoc=True)

    # test_gpr_vfe_probit(nat_param=True, stoc=False)
    # test_gpr_vfe_probit(nat_param=True, stoc=True)
    # test_gpr_vfe_probit(nat_param=False, stoc=False)
    # test_gpr_vfe_probit(nat_param=False, stoc=True)

    # plot_gpr_vfe_probit_stochastic()
    # plot_gpr_vfe_gaussian_stochastic()

    # TODO: Deep GP, nat param, different prop mode
    # test_dgpr_vfe_gaussian(nat_param=True, stoc=False)
    # test_dgpr_vfe_gaussian(nat_param=True, stoc=True)
    # # test_dgpr_vfe_gaussian(nat_param=False, stoc=False)
    # # test_dgpr_vfe_gaussian(nat_param=False, stoc=True)

    # test_dgpr_vfe_probit(nat_param=True, stoc=False)
    # test_dgpr_vfe_probit(nat_param=True, stoc=True)
    # # test_dgpr_vfe_probit(nat_param=False, stoc=False)
    # # test_dgpr_vfe_probit(nat_param=False, stoc=True)

    # plot_dgpr_vfe_probit_stochastic()
    # plot_dgpr_vfe_gaussian_stochastic()

    # TODO: Deep GP with hidden, nat param, different prop mode
    # test_dgprh_vfe_gaussian(nat_param=True, stoc=False)
    # test_dgprh_vfe_gaussian(nat_param=True, stoc=True)
    # # test_dgprh_vfe_gaussian(nat_param=False, stoc=False)
    # # test_dgprh_vfe_gaussian(nat_param=False, stoc=True)

    # test_dgprh_vfe_probit(nat_param=True, stoc=False)
    # test_dgprh_vfe_probit(nat_param=True, stoc=True)
    # # test_dgprh_vfe_probit(nat_param=False, stoc=False)
    # # test_dgprh_vfe_probit(nat_param=False, stoc=True)

    # TODO: GPSSM, prop MC
    # test_gpssm_linear_vfe_gaussian_kink(nat_param=True, stoc=False, prop_mode=PROP_MM)
    # test_gpssm_linear_vfe_gaussian_kink(nat_param=True, stoc=True, prop_mode=PROP_MM)
    # test_gpssm_linear_vfe_gaussian_kink(nat_param=False, stoc=False, prop_mode=PROP_MM)
    # test_gpssm_linear_vfe_gaussian_kink(nat_param=False, stoc=True, prop_mode=PROP_MM)

    # test_gpssm_linear_vfe_gaussian_kink(nat_param=True, stoc=False, prop_mode=PROP_MC)
    # test_gpssm_linear_vfe_gaussian_kink(nat_param=True, stoc=True, prop_mode=PROP_MC)
    # test_gpssm_linear_vfe_gaussian_kink(nat_param=False, stoc=False, prop_mode=PROP_MC)
    # test_gpssm_linear_vfe_gaussian_kink(nat_param=False, stoc=True, prop_mode=PROP_MC)

    # test_gpssm_gp_vfe_gaussian_kink(nat_param=True, stoc=False, prop_mode=PROP_MM)
    # test_gpssm_gp_vfe_gaussian_kink(nat_param=True, stoc=True, prop_mode=PROP_MM)
    # test_gpssm_gp_vfe_gaussian_kink(nat_param=False, stoc=False, prop_mode=PROP_MM)
    # test_gpssm_gp_vfe_gaussian_kink(nat_param=False, stoc=True, prop_mode=PROP_MM)

    # test_gpssm_gp_vfe_gaussian_kink(nat_param=True, stoc=False, prop_mode=PROP_MC)
    # test_gpssm_gp_vfe_gaussian_kink(nat_param=True, stoc=True, prop_mode=PROP_MC)
    # test_gpssm_gp_vfe_gaussian_kink(nat_param=False, stoc=False, prop_mode=PROP_MC)
    # test_gpssm_gp_vfe_gaussian_kink(nat_param=False, stoc=True, prop_mode=PROP_MC)

    # plot_gpssm_linear_vfe_gaussian_stochastic()
