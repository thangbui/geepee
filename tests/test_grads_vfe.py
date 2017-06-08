import numpy as np
from scipy.optimize import check_grad
import copy
import pdb
import pprint
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from .context import vfe
from .context import flatten_dict, unflatten_dict
from .context import PROP_MC, PROP_MM

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(0)


def objective(params, params_args, obj, idxs, alpha):
    params_dict = unflatten_dict(params, params_args)
    f, grad_dict = obj.objective_function(
        params_dict, idxs)
    return f


def gradient(params, params_args, obj, idxs, alpha):
    params_dict = unflatten_dict(params, params_args)
    f, grad_dict = obj.objective_function(
        params_dict, idxs, alpha=alpha)
    g, _ = flatten_dict(grad_dict)
    return g


def test_gpr_collapsed_vfe_gaussian():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 4
    D = 2
    Q = 1
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    # params  tied
    model = vfe.SGPR_collapsed(x_train, y_train, M)

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, None, alpha=alpha)
    pp.pprint(logZ)
    pp.pprint(params)

    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = model.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = model.objective_function(
            params1, None, alpha=alpha)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = model.objective_function(
            params2, None, alpha=alpha)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = model.objective_function(
        params1, None, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = model.objective_function(
        params2, None, alpha=alpha)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = model.objective_function(
        params1, None, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = model.objective_function(
        params2, None, alpha=alpha)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = model.Din
    M_i = model.M
    Dout_i = model.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            logZ1, grad1 = model.objective_function(
                params1, None, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = model.objective_function(
                params2, None, alpha=alpha)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))


def test_gpr_collapsed_vfe_gaussian_scipy():
    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 4
    D = 2
    Q = 1
    x_train = np.random.randn(N_train, D)
    y_train = np.random.randn(N_train, Q)
    # params  tied
    model = vfe.SGPR_collapsed(x_train, y_train, M)

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, None, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, None, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec,
                         params_args, model, None, alpha))


def test_gpr_vfe_gaussian():

    # generate some datapoints for testing
    N_train = 20
    alpha = 0.5
    M = 10
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model = vfe.SGPR(x_train, y_train, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, N_train)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = model.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = model.objective_function(
            params1, N_train)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = model.objective_function(
            params2, N_train)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = model.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = model.objective_function(
        params2, N_train)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = model.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = model.objective_function(
        params2, N_train)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = model.Din
    M_i = model.M
    Dout_i = model.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))


def test_gpr_vfe_gaussian_optimised():

    # generate some datapoints for testing
    N_train = 20
    M = 10
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model = vfe.SGPR(x_train, y_train, M, lik='Gaussian')
    model.optimise(method='L-BFGS-B', maxiter=1000)
    init_params = model.get_hypers()
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, N_train)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = model.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = model.objective_function(
            params1, N_train)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = model.objective_function(
            params2, N_train)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = model.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = model.objective_function(
        params2, N_train)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = model.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = model.objective_function(
        params2, N_train)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = model.Din
    M_i = model.M
    Dout_i = model.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))


def test_gpr_vfe_probit():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 3
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = vfe.SGPR(x_train, y_train, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, N_train)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = model.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = model.objective_function(
            params1, N_train)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = model.objective_function(
            params2, N_train)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = model.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = model.objective_function(
        params2, N_train)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad zu
    Din_i = model.Din
    M_i = model.M
    Dout_i = model.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))


def test_gpr_vfe_gaussian_stochastic():

    # generate some datapoints for testing
    N_train = 20
    alpha = 0.5
    M = 10
    D = 2
    Q = 3
    mb_size = M
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    model = vfe.SGPR(x_train, y_train, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    np.random.seed(0)
    logZ, grad_all = model.objective_function(params, mb_size)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = model.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        np.random.seed(0)
        logZ1, grad1 = model.objective_function(
            params1, mb_size)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        np.random.seed(0)
        logZ2, grad2 = model.objective_function(
            params2, mb_size)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    np.random.seed(0)
    logZ1, grad1 = model.objective_function(
        params1, mb_size)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    np.random.seed(0)
    logZ2, grad2 = model.objective_function(
        params2, mb_size)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    np.random.seed(0)
    logZ1, grad1 = model.objective_function(
        params1, mb_size)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    np.random.seed(0)
    logZ2, grad2 = model.objective_function(
        params2, mb_size)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = model.Din
    M_i = model.M
    Dout_i = model.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            np.random.seed(0)
            logZ1, grad1 = model.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            np.random.seed(0)
            logZ2, grad2 = model.objective_function(
                params2, mb_size)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            np.random.seed(0)
            logZ1, grad1 = model.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            np.random.seed(0)
            logZ2, grad2 = model.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            np.random.seed(0)
            logZ1, grad1 = model.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            np.random.seed(0)
            logZ2, grad2 = model.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))


def test_gpr_vfe_probit_stochastic():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 3
    mb_size = M
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = vfe.SGPR(x_train, y_train, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    np.random.seed(0)
    logZ, grad_all = model.objective_function(params, mb_size)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = model.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        np.random.seed(0)
        logZ1, grad1 = model.objective_function(
            params1, mb_size)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        np.random.seed(0)
        logZ2, grad2 = model.objective_function(
            params2, mb_size)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    np.random.seed(0)
    logZ1, grad1 = model.objective_function(
        params1, mb_size)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    np.random.seed(0)
    logZ2, grad2 = model.objective_function(
        params2, mb_size)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad zu
    Din_i = model.Din
    M_i = model.M
    Dout_i = model.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            np.random.seed(0)
            logZ1, grad1 = model.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            np.random.seed(0)
            logZ2, grad2 = model.objective_function(
                params2, mb_size)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            np.random.seed(0)
            logZ1, grad1 = model.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            np.random.seed(0)
            logZ2, grad2 = model.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            np.random.seed(0)
            logZ1, grad1 = model.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            np.random.seed(0)
            logZ2, grad2 = model.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))


def plot_gpr_vfe_gaussian_stochastic():

    # generate some datapoints for testing
    N_train = 2000
    alpha = 0.5
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
                params, no_points)[0]
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

    # generate some datapoints for testing
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
                params, no_points)[0]
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


def test_gpr_vfe_gaussian_scipy():
    # generate some datapoints for testing
    N_train = 20
    alpha = 0.5
    M = 10
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = np.random.randn(N_train, Q)
    model = vfe.SGPR(x_train, y_train, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, N_train)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, N_train, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec,
                         params_args, model, N_train, alpha))


def test_gpr_vfe_probit_scipy():
    # generate some datapoints for testing
    N_train = 10
    alpha = 0.5
    M = 5
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = vfe.SGPR(x_train, y_train, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, N_train)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, N_train, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec,
                         params_args, model, N_train, alpha))


def test_gplvm_vfe_gaussian(nat_param=True):

    # generate some datapoints for testing
    N_train = 5
    M = 4
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)
    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)
    
    # pdb.set_trace()
    # x = np.random.randn(N_train, D)
    # init_params['x2'] = np.log(1e6*np.ones_like(init_params['x2'])) / 2
    # init_params['x1'] = x * (1 + 1e6*np.ones_like(init_params['x2']))
    # lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian', prior_mean=x, prior_var=0.000001)
    # params = init_params.copy()
    # logZ, grad_all = lvm.objective_function(params, N_train)
    # reg = vfe.SGPR(x, y_train, M, lik='Gaussian')
    # reg_logZ = reg.objective_function(init_params, N_train)
    # pdb.set_trace()

    params = init_params.copy()
    logZ, grad_all = lvm.objective_function(
        params, N_train)
    pp.pprint(logZ)
    pp.pprint(params)

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = lvm.objective_function(
            params1, N_train)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = lvm.objective_function(
            params2, N_train)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, N_train)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, N_train)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = lvm.Din
    M_i = lvm.M
    Dout_i = lvm.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd) / dx2_nd))


def test_gplvm_vfe_gaussian_cluster():

    # generate some datapoints for testing
    import GPy
    print "creating dataset..."
    N = 20
    k1 = GPy.kern.RBF(5, variance=1, lengthscale=1. /
                      np.random.dirichlet(np.r_[10, 10, 10, 0.1, 0.1]), ARD=True)
    k2 = GPy.kern.RBF(5, variance=1, lengthscale=1. /
                      np.random.dirichlet(np.r_[10, 0.1, 10, 0.1, 10]), ARD=True)
    k3 = GPy.kern.RBF(5, variance=1, lengthscale=1. /
                      np.random.dirichlet(np.r_[0.1, 0.1, 10, 10, 10]), ARD=True)
    X = np.random.normal(0, 1, (N, 5))
    A = np.random.multivariate_normal(np.zeros(N), k1.K(X), 10).T
    B = np.random.multivariate_normal(np.zeros(N), k2.K(X), 10).T
    C = np.random.multivariate_normal(np.zeros(N), k3.K(X), 10).T

    Y = np.vstack((A, B, C))
    labels = np.hstack((np.zeros(A.shape[0]), np.ones(
        B.shape[0]), np.ones(C.shape[0]) * 2))

    N_train = Y.shape[0]
    M = 20
    D = 3
    y_train = Y
    lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    logZ, grad_all = lvm.objective_function(
        params, N_train)
    pp.pprint(logZ)
    pp.pprint(params)

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = lvm.objective_function(
            params1, N_train)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = lvm.objective_function(
            params2, N_train)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, N_train)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, N_train)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = lvm.Din
    M_i = lvm.M
    Dout_i = lvm.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd) / dx2_nd))


def test_gplvm_vfe_probit():

    # generate some datapoints for testing
    N_train = 5
    M = 3
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    lvm = vfe.SGPLVM(y_train, D, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    logZ, grad_all = lvm.objective_function(
        params, N_train)
    pp.pprint(logZ)

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = lvm.objective_function(
            params1, N_train)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = lvm.objective_function(
            params2, N_train)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, N_train)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad zu
    Din_i = lvm.Din
    M_i = lvm.M
    Dout_i = lvm.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd) / dx2_nd))


def test_gplvm_vfe_gaussian_MC():

    # generate some datapoints for testing
    N_train = 5
    M = 3
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    np.random.seed(42)
    logZ, grad_all = lvm.objective_function(
        params, N_train, prop_mode=PROP_MC)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        np.random.seed(42)
        logZ1, grad1 = lvm.objective_function(
            params1, N_train, prop_mode=PROP_MC)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        np.random.seed(42)
        logZ2, grad2 = lvm.objective_function(
            params2, N_train, prop_mode=PROP_MC)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    np.random.seed(42)
    logZ1, grad1 = lvm.objective_function(
        params1, N_train, prop_mode=PROP_MC)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    np.random.seed(42)
    logZ2, grad2 = lvm.objective_function(
        params2, N_train, prop_mode=PROP_MC)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    np.random.seed(42)
    logZ1, grad1 = lvm.objective_function(
        params1, N_train, prop_mode=PROP_MC)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    np.random.seed(42)
    logZ2, grad2 = lvm.objective_function(
        params2, N_train, prop_mode=PROP_MC)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = lvm.Din
    M_i = lvm.M
    Dout_i = lvm.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd) / dx2_nd))


def test_gplvm_vfe_probit_MC():

    # generate some datapoints for testing
    N_train = 5
    M = 3
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    lvm = vfe.SGPLVM(y_train, D, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    np.random.seed(42)
    logZ, grad_all = lvm.objective_function(
        params, N_train, prop_mode=PROP_MC)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        np.random.seed(42)
        logZ1, grad1 = lvm.objective_function(
            params1, N_train, prop_mode=PROP_MC)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        np.random.seed(42)
        logZ2, grad2 = lvm.objective_function(
            params2, N_train, prop_mode=PROP_MC)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    np.random.seed(42)
    logZ1, grad1 = lvm.objective_function(
        params1, N_train, prop_mode=PROP_MC)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    np.random.seed(42)
    logZ2, grad2 = lvm.objective_function(
        params2, N_train, prop_mode=PROP_MC)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad zu
    Din_i = lvm.Din
    M_i = lvm.M
    Dout_i = lvm.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, N_train, prop_mode=PROP_MC)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, N_train, prop_mode=PROP_MC)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd) / dx2_nd))


def test_gplvm_vfe_gaussian_stochastic(nat_param=True):

    # generate some datapoints for testing
    N_train = 10
    M = 5
    D = 2
    Q = 3
    mb_size = 3
    y_train = np.random.randn(N_train, Q)
    lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    np.random.seed(100)
    logZ, grad_all = lvm.objective_function(
        params, mb_size)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        np.random.seed(100)
        logZ1, grad1 = lvm.objective_function(
            params1, mb_size)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        np.random.seed(100)
        logZ2, grad2 = lvm.objective_function(
            params2, mb_size)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    np.random.seed(100)
    logZ1, grad1 = lvm.objective_function(
        params1, mb_size)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    np.random.seed(100)
    logZ2, grad2 = lvm.objective_function(
        params2, mb_size)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    np.random.seed(100)
    logZ1, grad1 = lvm.objective_function(
        params1, mb_size)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    np.random.seed(100)
    logZ2, grad2 = lvm.objective_function(
        params2, mb_size)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    Din_i = lvm.Din
    M_i = lvm.M
    Dout_i = lvm.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd) / dx2_nd))


def test_gplvm_vfe_probit_stochastic():

    # generate some datapoints for testing
    N_train = 5
    M = 3
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    lvm = vfe.SGPLVM(y_train, D, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    np.random.seed(100)
    logZ, grad_all = lvm.objective_function(
        params, mb_size)
    pp.pprint(logZ)

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        np.random.seed(100)
        logZ1, grad1 = lvm.objective_function(
            params1, mb_size)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        np.random.seed(100)
        logZ2, grad2 = lvm.objective_function(
            params2, mb_size)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d] - dls_id) / dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    np.random.seed(100)
    logZ1, grad1 = lvm.objective_function(
        params1, mb_size)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    np.random.seed(100)
    logZ2, grad2 = lvm.objective_function(
        params2, mb_size)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad zu
    Din_i = lvm.Din
    M_i = lvm.M
    Dout_i = lvm.Dout
    for m in range(M_i):
        for k in range(Din_i):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1['zu']
            eps1[m, k] = eps
            params1['zu'] = params1['zu'] + eps1
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j] - dR_id) / dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j] - dR_id) / dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            np.random.seed(100)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            np.random.seed(100)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd) / dx2_nd))


def plot_gplvm_vfe_gaussian_stochastic():

    # generate some datapoints for testing
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
                params, no_points)[0]
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

    # generate some datapoints for testing
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
                params, no_points)[0]
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


def test_gplvm_vfe_gaussian_scipy(nat_param=True):
    # generate some datapoints for testing
    N_train = 5
    M = 4
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian', nat_param=nat_param)

    # init hypers, inducing points and q(u) params
    init_params_dict = lvm.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = lvm.objective_function(
        params, N_train)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, lvm, N_train, None)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec,
                         params_args, lvm, N_train, None))


def test_gplvm_vfe_probit_scipy():
    # generate some datapoints for testing
    N_train = 5
    M = 4
    D = 2
    Q = 3
    y_train = 2 * np.random.randint(0, 2, size=(N_train, Q)) - 1
    lvm = vfe.SGPLVM(y_train, D, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params_dict = lvm.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = lvm.objective_function(
        params, N_train)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, lvm, N_train, None)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec,
                         params_args, lvm, N_train, None))


def test_gpssm_linear_vfe_gaussian_kink_MM(nat_param=True):
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

    N_train = 50
    process_noise = 0.2
    obs_noise = 0.1
    alpha = 0.5
    M = 4
    Q = 1
    D = 1
    (xtrue, x, y) = kink(N_train, process_noise, obs_noise)
    y_train = np.reshape(y, [y.shape[0], 1])
    lvm = vfe.SGPSSM(y_train, Q, M, lik='Gaussian', gp_emi=False, nat_param=nat_param)
    
    lvm.optimise(
        method='adam', maxiter=500, adam_lr=0.08)
    params = lvm.get_hypers()

    # # init hypers, inducing points and q(u) params
    # init_params = lvm.init_hypers(y_train)
    # params = init_params.copy()

    logZ, grad_all = lvm.objective_function(
        params, N_train)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    name = 'ls_dynamic'
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1[name][d] = params1[name][d] + eps
        logZ1, grad1 = lvm.objective_function(
            params1, N_train)

        params2 = copy.deepcopy(params)
        params2[name][d] = params2[name][d] - eps
        logZ2, grad2 = lvm.objective_function(
            params2, N_train)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all[name][d], dls_id, (grad_all[name][d] - dls_id) / dls_id))

    # check grad sf
    name = 'sf_dynamic'
    params1 = copy.deepcopy(params)
    params1[name] = params1[name] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2[name] = params2[name] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, N_train)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all[name], dsf_i, (grad_all[name] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, N_train)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, N_train)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    name = 'zu_dynamic'
    for m in range(M):
        for k in range(Q):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1[name]
            eps1[m, k] = eps
            params1[name] = params1[name] + eps1
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2[name] = params2[name] - eps1
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all[name][m, k], dzu_id, (grad_all[name][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    Dout = lvm.dyn_layer.Dout
    name = 'eta1_R_dynamic'
    for d in range(Dout):
        for j in range(M * (M + 1) / 2):
            params1 = copy.deepcopy(params)
            params1[name][d][j, ] = params1[name][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2[name][d][j, ] = params2[name][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j] - dR_id) / dR_id))

    # check grad theta_2
    name = 'eta2_dynamic'
    for d in range(Dout):
        for j in range(M):
            params1 = copy.deepcopy(params)
            params1[name][d][j, ] = params1[name][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2[name][d][j, ] = params2[name][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j] - dR_id) / dR_id))

    keys = ['x_factor_1', 'x_factor_2']
    for key in keys:
        # check grad x1
        for n in range(N_train):
            for d in range(Q):
                params1 = copy.deepcopy(params)
                params1[key][n, d] = params1[key][n, d] + eps
                logZ1, grad1 = lvm.objective_function(
                    params1, N_train)
                params2 = copy.deepcopy(params)
                params2[key][n, d] = params2[key][n, d] - eps
                logZ2, grad2 = lvm.objective_function(
                    params2, N_train)

                dx1_nd = (logZ1 - logZ2) / eps / 2
                print('%s n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                      % (key, n, d, grad_all[key][n, d], dx1_nd, (grad_all[key][n, d] - dx1_nd) / dx1_nd))

    # check grad C
    for d in range(D):
        for q in range(Q):
            params1 = copy.deepcopy(params)
            params1['C_emission'][d, q] = params1['C_emission'][d, q] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, N_train)
            params2 = copy.deepcopy(params)
            params2['C_emission'][d, q] = params2['C_emission'][d, q] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, N_train)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('C d=%d, q=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, q, grad_all['C_emission'][d, q], dx1_nd, (grad_all['C_emission'][d, q] - dx1_nd) / dx1_nd))

    # check grad R
    for d in range(D):
        params1 = copy.deepcopy(params)
        params1['R_emission'][d] = params1['R_emission'][d] + eps
        logZ1, grad1 = lvm.objective_function(
            params1, N_train)
        params2 = copy.deepcopy(params)
        params2['R_emission'][d] = params2['R_emission'][d] - eps
        logZ2, grad2 = lvm.objective_function(
            params2, N_train)

        dx1_nd = (logZ1 - logZ2) / eps / 2
        print('R d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['R_emission'][d], dx1_nd, (grad_all['R_emission'][d] - dx1_nd) / dx1_nd))


def test_gpssm_linear_vfe_gaussian_kink_MM_stochastic(nat_param=True):
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

    N_train = 50
    process_noise = 0.2
    obs_noise = 0.1
    alpha = 0.5
    M = 4
    Q = 1
    D = 1
    mb_size = M
    (xtrue, x, y) = kink(N_train, process_noise, obs_noise)
    y_train = np.reshape(y, [y.shape[0], 1])
    lvm = vfe.SGPSSM(y_train, Q, M, lik='Gaussian', gp_emi=False, nat_param=nat_param)
    
    # lvm.optimise(
    #     method='adam', maxiter=500, adam_lr=0.08)
    # params = lvm.get_hypers()

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)
    params = init_params.copy()

    np.random.seed(42)
    logZ, grad_all = lvm.objective_function(
        params, mb_size, prop_mode=PROP_MM)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    name = 'ls_dynamic'
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1[name][d] = params1[name][d] + eps
        np.random.seed(42)
        logZ1, grad1 = lvm.objective_function(
            params1, mb_size, prop_mode=PROP_MM)

        params2 = copy.deepcopy(params)
        params2[name][d] = params2[name][d] - eps
        np.random.seed(42)
        logZ2, grad2 = lvm.objective_function(
            params2, mb_size, prop_mode=PROP_MM)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all[name][d], dls_id, (grad_all[name][d] - dls_id) / dls_id))

    # check grad sf
    name = 'sf_dynamic'
    params1 = copy.deepcopy(params)
    params1[name] = params1[name] + eps
    np.random.seed(42)
    logZ1, grad1 = lvm.objective_function(
        params1, mb_size, prop_mode=PROP_MM)
    params2 = copy.deepcopy(params)
    params2[name] = params2[name] - eps
    np.random.seed(42)
    logZ2, grad2 = lvm.objective_function(
        params2, mb_size, prop_mode=PROP_MM)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print('sf computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all[name], dsf_i, (grad_all[name] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    np.random.seed(42)
    logZ1, grad1 = lvm.objective_function(
        params1, mb_size, prop_mode=PROP_MM)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    np.random.seed(42)
    logZ2, grad2 = lvm.objective_function(
        params2, mb_size, prop_mode=PROP_MM)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print('sn computed=%.5f, numerical=%.5f, diff=%.5f'
          % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))

    # check grad zu
    name = 'zu_dynamic'
    for m in range(M):
        for k in range(Q):
            params1 = copy.deepcopy(params)
            eps1 = 0.0 * params1[name]
            eps1[m, k] = eps
            params1[name] = params1[name] + eps1
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size, prop_mode=PROP_MM)
            params2 = copy.deepcopy(params)
            params2[name] = params2[name] - eps1
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size, prop_mode=PROP_MM)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (m, k, grad_all[name][m, k], dzu_id, (grad_all[name][m, k] - dzu_id) / dzu_id))

    # check grad theta_1
    Dout = lvm.dyn_layer.Dout
    name = 'eta1_R_dynamic'
    for d in range(Dout):
        for j in range(M * (M + 1) / 2):
            params1 = copy.deepcopy(params)
            params1[name][d][j, ] = params1[name][d][j, ] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size, prop_mode=PROP_MM)
            params2 = copy.deepcopy(params)
            params2[name][d][j, ] = params2[name][d][j, ] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size, prop_mode=PROP_MM)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j] - dR_id) / dR_id))

    # check grad theta_2
    name = 'eta2_dynamic'
    for d in range(Dout):
        for j in range(M):
            params1 = copy.deepcopy(params)
            params1[name][d][j, ] = params1[name][d][j, ] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size, prop_mode=PROP_MM)
            params2 = copy.deepcopy(params)
            params2[name][d][j, ] = params2[name][d][j, ] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size, prop_mode=PROP_MM)

            dR_id = (logZ1 - logZ2) / eps / 2
            print('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j] - dR_id) / dR_id))

    keys = ['x_factor_1', 'x_factor_2']
    for key in keys:
        # check grad x1
        for n in range(N_train):
            for d in range(Q):
                params1 = copy.deepcopy(params)
                params1[key][n, d] = params1[key][n, d] + eps
                np.random.seed(42)
                logZ1, grad1 = lvm.objective_function(
                    params1, mb_size, prop_mode=PROP_MM)
                params2 = copy.deepcopy(params)
                params2[key][n, d] = params2[key][n, d] - eps
                np.random.seed(42)
                logZ2, grad2 = lvm.objective_function(
                    params2, mb_size, prop_mode=PROP_MM)

                dx1_nd = (logZ1 - logZ2) / eps / 2
                print('%s n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                      % (key, n, d, grad_all[key][n, d], dx1_nd, (grad_all[key][n, d] - dx1_nd) / dx1_nd))

    # check grad C
    for d in range(D):
        for q in range(Q):
            params1 = copy.deepcopy(params)
            params1['C_emission'][d, q] = params1['C_emission'][d, q] + eps
            np.random.seed(42)
            logZ1, grad1 = lvm.objective_function(
                params1, mb_size, prop_mode=PROP_MM)
            params2 = copy.deepcopy(params)
            params2['C_emission'][d, q] = params2['C_emission'][d, q] - eps
            np.random.seed(42)
            logZ2, grad2 = lvm.objective_function(
                params2, mb_size, prop_mode=PROP_MM)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print('C d=%d, q=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (d, q, grad_all['C_emission'][d, q], dx1_nd, (grad_all['C_emission'][d, q] - dx1_nd) / dx1_nd))

    # check grad R
    for d in range(D):
        params1 = copy.deepcopy(params)
        params1['R_emission'][d] = params1['R_emission'][d] + eps
        np.random.seed(42)
        logZ1, grad1 = lvm.objective_function(
            params1, mb_size, prop_mode=PROP_MM)
        params2 = copy.deepcopy(params)
        params2['R_emission'][d] = params2['R_emission'][d] - eps
        np.random.seed(42)
        logZ2, grad2 = lvm.objective_function(
            params2, mb_size, prop_mode=PROP_MM)

        dx1_nd = (logZ1 - logZ2) / eps / 2
        print('R d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (d, grad_all['R_emission'][d], dx1_nd, (grad_all['R_emission'][d] - dx1_nd) / dx1_nd))


if __name__ == '__main__':
    # for i in range(10):
    #     test_gpr_collapsed_vfe_gaussian()
    #     test_gpr_collapsed_vfe_gaussian_scipy()

    # test_gpr_vfe_gaussian()
    # test_gpr_vfe_gaussian_optimised()
    # test_gpr_vfe_gaussian_scipy()
    # test_gpr_vfe_gaussian_stochastic()
    # plot_gpr_vfe_gaussian_stochastic()

    test_gpr_vfe_probit()
    test_gpr_vfe_probit_scipy()
    test_gpr_vfe_probit_stochastic()
    # plot_gpr_vfe_probit_stochastic()

    # test_gplvm_vfe_gaussian(True)
    # test_gplvm_vfe_gaussian(False)
    # test_gplvm_vfe_gaussian_cluster()
    # test_gplvm_vfe_gaussian_scipy(True)
    # test_gplvm_vfe_gaussian_scipy(False)
    # test_gplvm_vfe_gaussian_MC()
    # test_gplvm_vfe_gaussian_stochastic(True)
    # test_gplvm_vfe_gaussian_stochastic(False)
    # plot_gplvm_vfe_gaussian_stochastic()

    # test_gplvm_vfe_probit()
    # test_gplvm_vfe_probit_scipy()
    # test_gplvm_vfe_probit_MC()
    # test_gplvm_vfe_probit_stochastic()
    # plot_gplvm_vfe_probit_stochastic()

    # test_gpssm_linear_vfe_gaussian_kink_MM(True)
    # test_gpssm_linear_vfe_gaussian_kink_MM_stochastic(True)
    # test_gpssm_linear_vfe_gaussian_kink_MM(False)
    # test_gpssm_linear_vfe_gaussian_kink_MM_stochastic(False)
