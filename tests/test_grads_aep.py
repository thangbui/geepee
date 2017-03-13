import numpy as np
from scipy.optimize import check_grad
import copy
import pdb
import pprint
import os

from .context import aep
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


def test_gplvm_aep_gaussian():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 10
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    # params  tied
    lvm = aep.SGPLVM(y_train, D, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    logZ, grad_all = lvm.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)
    pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = lvm.objective_function(
            params1, idxs, alpha=alpha)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = lvm.objective_function(
            params2, idxs, alpha=alpha)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print ('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
            % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d]-dls_id)/dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, idxs, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, idxs, alpha=alpha)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print ('sf computed=%.5f, numerical=%.5f, diff=%.5f' 
        % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, idxs, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, idxs, alpha=alpha)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print ('sn computed=%.5f, numerical=%.5f, diff=%.5f' 
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
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print ('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k]-dzu_id)/dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j]-dR_id)/dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j]-dR_id)/dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print ('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d] - dx1_nd) / dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print ('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d] - dx2_nd)/dx2_nd))


def test_gplvm_aep_probit():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 3
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    y_train = 2*np.random.randint(0, 2, size=(N_train, Q)) - 1
    # params  tied
    lvm = aep.SGPLVM(y_train, D, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = lvm.init_hypers(y_train)

    params = init_params.copy()
    logZ, grad_all = lvm.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    eps = 1e-5
    # check grad ls
    Din_i = lvm.Din
    for d in range(Din_i):
        params1 = copy.deepcopy(params)
        params1['ls'][d] = params1['ls'][d] + eps
        logZ1, grad1 = lvm.objective_function(
            params1, idxs, alpha=alpha)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = lvm.objective_function(
            params2, idxs, alpha=alpha)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print ('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
            % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d]-dls_id)/dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = lvm.objective_function(
        params1, idxs, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = lvm.objective_function(
        params2, idxs, alpha=alpha)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print ('sf computed=%.5f, numerical=%.5f, diff=%.5f' 
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
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print ('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k]-dzu_id)/dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j]-dR_id)/dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j]-dR_id)/dR_id))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x1'][n, d] = params1['x1'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['x1'][n, d] = params2['x1'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dx1_nd = (logZ1 - logZ2) / eps / 2
            print ('x1 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (n, d, grad_all['x1'][n, d], dx1_nd, (grad_all['x1'][n, d]-dx1_nd)/dx1_nd))

    # check grad x1
    Din = lvm.Din
    for n in range(N_train):
        for d in range(Din):
            params1 = copy.deepcopy(params)
            params1['x2'][n, d] = params1['x2'][n, d] + eps
            logZ1, grad1 = lvm.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['x2'][n, d] = params2['x2'][n, d] - eps
            logZ2, grad2 = lvm.objective_function(
                params2, idxs, alpha=alpha)

            dx2_nd = (logZ1 - logZ2) / eps / 2
            print ('x2 n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (n, d, grad_all['x2'][n, d], dx2_nd, (grad_all['x2'][n, d]-dx2_nd)/dx2_nd))


def test_gplvm_aep_gaussian_scipy():
    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 10
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    # params  tied
    lvm = aep.SGPLVM(y_train, D, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params_dict = lvm.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = lvm.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, lvm, idxs, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec, params_args, lvm, idxs, alpha))


def test_gplvm_aep_probit_scipy():
    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 10
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    y_train = 2*np.random.randint(0, 2, size=(N_train, Q)) - 1
    # params  tied
    lvm = aep.SGPLVM(y_train, D, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params_dict = lvm.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = lvm.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, lvm, idxs, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec, params_args, lvm, idxs, alpha))


def test_gpr_aep_gaussian():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 10
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    # params  tied
    model = aep.SGPR(x_train, y_train, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, idxs, alpha=alpha)
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
            params1, idxs, alpha=alpha)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = model.objective_function(
            params2, idxs, alpha=alpha)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print ('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
            % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d]-dls_id)/dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = model.objective_function(
        params1, idxs, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = model.objective_function(
        params2, idxs, alpha=alpha)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print ('sf computed=%.5f, numerical=%.5f, diff=%.5f' 
        % (grad_all['sf'], dsf_i, (grad_all['sf'] - dsf_i) / dsf_i))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = model.objective_function(
        params1, idxs, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = model.objective_function(
        params2, idxs, alpha=alpha)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print ('sn computed=%.5f, numerical=%.5f, diff=%.5f' 
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
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print ('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k]-dzu_id)/dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j]-dR_id)/dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j]-dR_id)/dR_id))


def test_gpr_aep_probit():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 3
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2*np.random.randint(0, 2, size=(N_train, Q)) - 1
    # params  tied
    model = aep.SGPR(x_train, y_train, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, idxs, alpha=alpha)
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
            params1, idxs, alpha=alpha)

        params2 = copy.deepcopy(params)
        params2['ls'][d] = params2['ls'][d] - eps
        logZ2, grad2 = model.objective_function(
            params2, idxs, alpha=alpha)

        dls_id = (logZ1 - logZ2) / eps / 2
        # print logZ1, logZ2
        print ('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
            % (d, grad_all['ls'][d], dls_id, (grad_all['ls'][d]-dls_id)/dls_id))

    # check grad sf
    params1 = copy.deepcopy(params)
    params1['sf'] = params1['sf'] + eps
    logZ1, grad1 = model.objective_function(
        params1, idxs, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sf'] = params2['sf'] - eps
    logZ2, grad2 = model.objective_function(
        params2, idxs, alpha=alpha)

    dsf_i = (logZ1 - logZ2) / eps / 2
    print ('sf computed=%.5f, numerical=%.5f, diff=%.5f' 
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
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['zu'] = params2['zu'] - eps1
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dzu_id = (logZ1 - logZ2) / eps / 2
            print ('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (m, k, grad_all['zu'][m, k], dzu_id, (grad_all['zu'][m, k]-dzu_id)/dzu_id))

    # check grad theta_1
    for d in range(Dout_i):
        for j in range(M_i * (M_i + 1) / 2):
            params1 = copy.deepcopy(params)
            params1['eta1_R'][d][j, ] = params1['eta1_R'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta1_R'][d][j, ] = params2['eta1_R'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta1_R'][d][j], dR_id, (grad_all['eta1_R'][d][j]-dR_id)/dR_id))

    # check grad theta_2
    for d in range(Dout_i):
        for j in range(M_i):
            params1 = copy.deepcopy(params)
            params1['eta2'][d][j, ] = params1['eta2'][d][j, ] + eps
            logZ1, grad1 = model.objective_function(
                params1, idxs, alpha=alpha)
            params2 = copy.deepcopy(params)
            params2['eta2'][d][j, ] = params2['eta2'][d][j, ] - eps
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dR_id = (logZ1 - logZ2) / eps / 2
            print ('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, j, grad_all['eta2'][d][j], dR_id, (grad_all['eta2'][d][j]-dR_id)/dR_id))


def test_gpr_aep_gaussian_scipy():
    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 10
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = np.random.randn(N_train, Q)
    # params  tied
    model = aep.SGPR(x_train, y_train, M, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, idxs, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec, params_args, model, idxs, alpha))


def test_gpr_aep_probit_scipy():
    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 10
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    x_train = np.random.randn(N_train, D)
    y_train = 2*np.random.randint(0, 2, size=(N_train, Q)) - 1
    # params  tied
    model = aep.SGPR(x_train, y_train, M, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, idxs, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec, params_args, model, idxs, alpha))



def test_dgpr_aep_gaussian():

    # generate some datapoints for testing
    N_train = 10
    alpha = 1
    M = 5
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    hidden_size = [3, 2]
    # params  tied
    model = aep.SDGPR(x_train, y_train, M, hidden_size, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, idxs, alpha=alpha)
    # pp.pprint(logZ)
    # pp.pprint(params)
    # pdb.set_trace()

    eps = 1e-5
    # check grad ls
    size = model.size
    Ms = model.Ms
    L = model.L
    for i in range(L):
        Din_i = size[i]
        M_i = Ms[i]
        Dout_i = size[i+1]
        suffix = '_%d' % (i)
        name = 'ls' + suffix
        for d in range(Din_i):
            params1 = copy.deepcopy(params)
            params1[name][d] = params1[name][d] + eps
            logZ1, grad1 = model.objective_function(
                params1, idxs, alpha=alpha)

            params2 = copy.deepcopy(params)
            params2[name][d] = params2[name][d] - eps
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dls_id = (logZ1 - logZ2) / eps / 2
            # print logZ1, logZ2
            print ('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, grad_all[name][d], dls_id, (grad_all[name][d]-dls_id)/dls_id))

        # check grad sf
        name = 'sf' + suffix
        params1 = copy.deepcopy(params)
        params1[name] = params1[name] + eps
        logZ1, grad1 = model.objective_function(
            params1, idxs, alpha=alpha)
        params2 = copy.deepcopy(params)
        params2[name] = params2[name] - eps
        logZ2, grad2 = model.objective_function(
            params2, idxs, alpha=alpha)

        dsf_i = (logZ1 - logZ2) / eps / 2
        print ('sf computed=%.5f, numerical=%.5f, diff=%.5f' 
            % (grad_all[name], dsf_i, (grad_all[name] - dsf_i) / dsf_i))

        # check grad zu
        name = 'zu' + suffix
        for m in range(M_i):
            for k in range(Din_i):
                params1 = copy.deepcopy(params)
                eps1 = 0.0 * params1[name]
                eps1[m, k] = eps
                params1[name] = params1[name] + eps1
                logZ1, grad1 = model.objective_function(
                    params1, idxs, alpha=alpha)
                params2 = copy.deepcopy(params)
                params2[name] = params2[name] - eps1
                logZ2, grad2 = model.objective_function(
                    params2, idxs, alpha=alpha)

                dzu_id = (logZ1 - logZ2) / eps / 2
                print ('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                    % (m, k, grad_all[name][m, k], dzu_id, (grad_all[name][m, k]-dzu_id)/dzu_id))

        # check grad theta_1
        name = 'eta1_R' + suffix
        for d in range(Dout_i):
            for j in range(M_i * (M_i + 1) / 2):
                params1 = copy.deepcopy(params)
                params1[name][d][j, ] = params1[name][d][j, ] + eps
                logZ1, grad1 = model.objective_function(
                    params1, idxs, alpha=alpha)
                params2 = copy.deepcopy(params)
                params2[name][d][j, ] = params2[name][d][j, ] - eps
                logZ2, grad2 = model.objective_function(
                    params2, idxs, alpha=alpha)

                dR_id = (logZ1 - logZ2) / eps / 2
                print ('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                    % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j]-dR_id)/dR_id))

        # check grad theta_2
        name = 'eta2' + suffix
        for d in range(Dout_i):
            for j in range(M_i):
                params1 = copy.deepcopy(params)
                params1[name][d][j, ] = params1[name][d][j, ] + eps
                logZ1, grad1 = model.objective_function(
                    params1, idxs, alpha=alpha)
                params2 = copy.deepcopy(params)
                params2[name][d][j, ] = params2[name][d][j, ] - eps
                logZ2, grad2 = model.objective_function(
                    params2, idxs, alpha=alpha)

                dR_id = (logZ1 - logZ2) / eps / 2
                print ('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                    % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j]-dR_id)/dR_id))

    # check grad sn
    params1 = copy.deepcopy(params)
    params1['sn'] = params1['sn'] + eps
    logZ1, grad1 = model.objective_function(
        params1, idxs, alpha=alpha)
    params2 = copy.deepcopy(params)
    params2['sn'] = params2['sn'] - eps
    logZ2, grad2 = model.objective_function(
        params2, idxs, alpha=alpha)

    dsn_i = (logZ1 - logZ2) / eps / 2
    print ('sn computed=%.5f, numerical=%.5f, diff=%.5f' 
        % (grad_all['sn'], dsn_i, (grad_all['sn'] - dsn_i) / dsn_i))


def test_dgpr_aep_probit():

    # generate some datapoints for testing
    N_train = 5
    alpha = 1
    M = 3
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    hidden_size = [3, 2]
    x_train = np.random.randn(N_train, D)
    y_train = 2*np.random.randint(0, 2, size=(N_train, Q)) - 1
    model = aep.SDGPR(x_train, y_train, M, hidden_size, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers(y_train)
    params = init_params.copy()
    logZ, grad_all = model.objective_function(params, idxs, alpha=alpha)
    # pp.pprint(logZ)
    # pp.pprint(params)

    eps = 1e-5
    # check grad ls
    size = model.size
    Ms = model.Ms
    L = model.L
    for i in range(L):
        Din_i = size[i]
        M_i = Ms[i]
        Dout_i = size[i+1]
        suffix = '_%d' % (i)
        name = 'ls' + suffix
        for d in range(Din_i):
            params1 = copy.deepcopy(params)
            params1[name][d] = params1[name][d] + eps
            logZ1, grad1 = model.objective_function(
                params1, idxs, alpha=alpha)

            params2 = copy.deepcopy(params)
            params2[name][d] = params2[name][d] - eps
            logZ2, grad2 = model.objective_function(
                params2, idxs, alpha=alpha)

            dls_id = (logZ1 - logZ2) / eps / 2
            # print logZ1, logZ2
            print ('ls d=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                % (d, grad_all[name][d], dls_id, (grad_all[name][d]-dls_id)/dls_id))

        # check grad sf
        name = 'sf' + suffix
        params1 = copy.deepcopy(params)
        params1[name] = params1[name] + eps
        logZ1, grad1 = model.objective_function(
            params1, idxs, alpha=alpha)
        params2 = copy.deepcopy(params)
        params2[name] = params2[name] - eps
        logZ2, grad2 = model.objective_function(
            params2, idxs, alpha=alpha)

        dsf_i = (logZ1 - logZ2) / eps / 2
        print ('sf computed=%.5f, numerical=%.5f, diff=%.5f' 
            % (grad_all[name], dsf_i, (grad_all[name] - dsf_i) / dsf_i))

        # check grad zu
        name = 'zu' + suffix
        for m in range(M_i):
            for k in range(Din_i):
                params1 = copy.deepcopy(params)
                eps1 = 0.0 * params1[name]
                eps1[m, k] = eps
                params1[name] = params1[name] + eps1
                logZ1, grad1 = model.objective_function(
                    params1, idxs, alpha=alpha)
                params2 = copy.deepcopy(params)
                params2[name] = params2[name] - eps1
                logZ2, grad2 = model.objective_function(
                    params2, idxs, alpha=alpha)

                dzu_id = (logZ1 - logZ2) / eps / 2
                print ('zu m=%d, k=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                    % (m, k, grad_all[name][m, k], dzu_id, (grad_all[name][m, k]-dzu_id)/dzu_id))

        # check grad theta_1
        name = 'eta1_R' + suffix
        for d in range(Dout_i):
            for j in range(M_i * (M_i + 1) / 2):
                params1 = copy.deepcopy(params)
                params1[name][d][j, ] = params1[name][d][j, ] + eps
                logZ1, grad1 = model.objective_function(
                    params1, idxs, alpha=alpha)
                params2 = copy.deepcopy(params)
                params2[name][d][j, ] = params2[name][d][j, ] - eps
                logZ2, grad2 = model.objective_function(
                    params2, idxs, alpha=alpha)

                dR_id = (logZ1 - logZ2) / eps / 2
                print ('eta1_R d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                    % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j]-dR_id)/dR_id))

        # check grad theta_2
        name = 'eta2' + suffix
        for d in range(Dout_i):
            for j in range(M_i):
                params1 = copy.deepcopy(params)
                params1[name][d][j, ] = params1[name][d][j, ] + eps
                logZ1, grad1 = model.objective_function(
                    params1, idxs, alpha=alpha)
                params2 = copy.deepcopy(params)
                params2[name][d][j, ] = params2[name][d][j, ] - eps
                logZ2, grad2 = model.objective_function(
                    params2, idxs, alpha=alpha)

                dR_id = (logZ1 - logZ2) / eps / 2
                print ('eta2 d=%d, j=%d, computed=%.5f, numerical=%.5f, diff=%.5f' 
                    % (d, j, grad_all[name][d][j], dR_id, (grad_all[name][d][j]-dR_id)/dR_id))


def test_dgpr_aep_gaussian_scipy():
    # generate some datapoints for testing
    N_train = 10
    alpha = 1
    M = 5
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    hidden_size = [3, 2]
    x_train = np.random.randn(N_train, D)
    y_train = np.random.randn(N_train, Q)
    
    model = aep.SDGPR(x_train, y_train, M, hidden_size, lik='Gaussian')

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers(y_train)
    # pdb.set_trace()
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, idxs, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec, params_args, model, idxs, alpha))


def test_dgpr_aep_probit_scipy():
    # generate some datapoints for testing
    N_train = 10
    alpha = 1
    M = 5
    idxs = np.arange(N_train)
    D = 2
    Q = 3
    hidden_size = [3, 2]
    x_train = np.random.randn(N_train, D)
    y_train = 2*np.random.randint(0, 2, size=(N_train, Q)) - 1
    # params  tied
    model = aep.SDGPR(x_train, y_train, M, hidden_size, lik='Probit')

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers(y_train)
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, idxs, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec, params_args, model, idxs, alpha))

if __name__ == '__main__':
    # test_gplvm_aep_gaussian()
    # test_gplvm_aep_probit()
    # test_gplvm_aep_gaussian_scipy()
    # test_gplvm_aep_probit_scipy()

    # test_gpr_aep_gaussian()
    # test_gpr_aep_probit()
    # test_gpr_aep_gaussian_scipy()
    # test_gpr_aep_probit_scipy()


    test_dgpr_aep_gaussian()
    test_dgpr_aep_probit()
    test_dgpr_aep_gaussian_scipy()
    test_dgpr_aep_probit_scipy()