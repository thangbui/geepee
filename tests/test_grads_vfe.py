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

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(0)


def objective(params, params_args, obj, idxs, alpha):
    params_dict = unflatten_dict(params, params_args)
    f, grad_dict = obj.objective_function(
        params_dict, idxs, alpha=alpha)
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
    init_params = model.init_hypers()
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
    init_params_dict = model.init_hypers()
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



if __name__ == '__main__':
    # for i in range(10):
    #     test_gpr_collapsed_vfe_gaussian()
    #     test_gpr_collapsed_vfe_gaussian_scipy()

    # test_gpr_vfe_gaussian()
    # test_gpr_vfe_gaussian_optimised()
    # test_gpr_vfe_gaussian_scipy()
    # test_gpr_vfe_gaussian_stochastic()
    plot_gpr_vfe_gaussian_stochastic()

    # test_gpr_vfe_probit()
    # test_gpr_vfe_probit_scipy()
    # test_gpr_vfe_probit_stochastic()
    # plot_gpr_vfe_probit_stochastic()


