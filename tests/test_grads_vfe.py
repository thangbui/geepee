import numpy as np
from scipy.optimize import check_grad
import copy
import pdb
import pprint
import os

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


def test_gpr_vfe_gaussian():

    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 4
    idxs = np.arange(N_train)
    D = 2
    Q = 1
    y_train = np.random.randn(N_train, Q)
    x_train = np.random.randn(N_train, D)
    # params  tied
    model = vfe.SGPR(x_train, y_train, M)

    # init hypers, inducing points and q(u) params
    init_params = model.init_hypers()
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


def test_gpr_vfe_gaussian_scipy():
    # generate some datapoints for testing
    N_train = 5
    alpha = 0.5
    M = 4
    idxs = np.arange(N_train)
    D = 2
    Q = 1
    x_train = np.random.randn(N_train, D)
    y_train = np.random.randn(N_train, Q)
    # params  tied
    model = vfe.SGPR(x_train, y_train, M)

    # init hypers, inducing points and q(u) params
    init_params_dict = model.init_hypers()
    init_params_vec, params_args = flatten_dict(init_params_dict)

    params = init_params_dict.copy()
    logZ, grad_all = model.objective_function(
        params, idxs, alpha=alpha)
    pp.pprint(logZ)

    logZ = objective(init_params_vec, params_args, model, idxs, alpha)
    pp.pprint(logZ)

    pp.pprint(check_grad(objective, gradient, init_params_vec, params_args, model, idxs, alpha))

if __name__ == '__main__':
    test_gpr_vfe_gaussian()
    test_gpr_vfe_gaussian_scipy()
    
