import numpy as np
from scipy.optimize import check_grad
import copy
import pdb
import pprint
import os

from .context import lik
from .context import flatten_dict, unflatten_dict

pp = pprint.PrettyPrinter(indent=4)
np.random.seed(0)


def test_gauss_emis_log_tilted():

    # generate some datapoints for testing
    N = 5
    alpha = 0.0005
    Dout = 3
    Din = 2
    y_train = np.random.randn(N, Dout)
    # params  tied
    model = lik.Gauss_Emis(y_train, Dout, Din)

    params = {'C': np.random.randn(Dout, Din),
              'R': np.random.randn(Dout)}
    mx = np.random.randn(N, Din)
    vx = np.random.rand(N, Din)
    scale = 1

    model.update_hypers(params)
    logZ, grad_input, grad_params = model.compute_emission_tilted(
        mx, vx, alpha, scale)

    eps = 1e-5
    # check grad C
    for a in range(Dout):
        for b in range(Din):
            params1 = copy.deepcopy(params)
            params1['C'][a, b] = params1['C'][a, b] + eps
            model.update_hypers(params1)
            logZ1, _, _ = model.compute_emission_tilted(
                mx, vx, alpha, scale)

            params2 = copy.deepcopy(params)
            params2['C'][a, b] = params2['C'][a, b] - eps
            model.update_hypers(params2)
            logZ2, _, _ = model.compute_emission_tilted(
                mx, vx, alpha, scale)

            dab = (logZ1 - logZ2) / eps / 2
            print('C a=%d, b=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (a, b, grad_params['C'][a, b], dab, (grad_params['C'][a, b] - dab) / dab))

    # check grad R
    for a in range(Dout):
        params1 = copy.deepcopy(params)
        params1['R'][a] = params1['R'][a] + eps
        model.update_hypers(params1)
        logZ1, _, _ = model.compute_emission_tilted(
            mx, vx, alpha, scale)

        params2 = copy.deepcopy(params)
        params2['R'][a] = params2['R'][a] - eps
        model.update_hypers(params2)
        logZ2, _, _ = model.compute_emission_tilted(
            mx, vx, alpha, scale)

        da = (logZ1 - logZ2) / eps / 2
        print('R a=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (a, grad_params['R'][a], da, (grad_params['R'][a] - da) / da))

    eps = 1e-5
    # check grad mx
    for a in range(N):
        for b in range(Din):
            mx1 = copy.deepcopy(mx)
            mx1[a, b] = mx1[a, b] + eps
            model.update_hypers(params)
            logZ1, _, _ = model.compute_emission_tilted(
                mx1, vx, alpha, scale)

            mx2 = copy.deepcopy(mx)
            mx2[a, b] = mx2[a, b] - eps
            model.update_hypers(params)
            logZ2, _, _ = model.compute_emission_tilted(
                mx2, vx, alpha, scale)

            dab = (logZ1 - logZ2) / eps / 2
            print('mx n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (a, b, grad_input['mx'][a, b], dab, (grad_input['mx'][a, b] - dab) / dab))

    # check grad mx
    for a in range(N):
        for b in range(Din):
            vx1 = copy.deepcopy(vx)
            vx1[a, b] = vx1[a, b] + eps
            model.update_hypers(params)
            logZ1, _, _ = model.compute_emission_tilted(
                mx, vx1, alpha, scale)

            vx2 = copy.deepcopy(vx)
            vx2[a, b] = vx2[a, b] - eps
            model.update_hypers(params)
            logZ2, _, _ = model.compute_emission_tilted(
                mx, vx2, alpha, scale)

            dab = (logZ1 - logZ2) / eps / 2
            print('vx n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (a, b, grad_input['vx'][a, b], dab, (grad_input['vx'][a, b] - dab) / dab))


def test_gauss_emis_log_lik():

    # generate some datapoints for testing
    N = 5
    Dout = 3
    Din = 2
    y_train = np.random.randn(N, Dout)
    # params  tied
    model = lik.Gauss_Emis(y_train, Dout, Din)

    params = {'C': np.random.randn(Dout, Din),
              'R': np.random.randn(Dout)}
    mx = np.random.randn(N, Din)
    vx = np.random.rand(N, Din)
    scale = 1

    model.update_hypers(params)
    logZ, grad_input, grad_params = model.compute_emission_log_lik_exp(
        mx, vx, scale)

    eps = 1e-5
    # check grad C
    for a in range(Dout):
        for b in range(Din):
            params1 = copy.deepcopy(params)
            params1['C'][a, b] = params1['C'][a, b] + eps
            model.update_hypers(params1)
            logZ1, _, _ = model.compute_emission_log_lik_exp(
                mx, vx, scale)

            params2 = copy.deepcopy(params)
            params2['C'][a, b] = params2['C'][a, b] - eps
            model.update_hypers(params2)
            logZ2, _, _ = model.compute_emission_log_lik_exp(
                mx, vx, scale)

            dab = (logZ1 - logZ2) / eps / 2
            print('C a=%d, b=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (a, b, grad_params['C'][a, b], dab, (grad_params['C'][a, b] - dab) / dab))

    # check grad R
    for a in range(Dout):
        params1 = copy.deepcopy(params)
        params1['R'][a] = params1['R'][a] + eps
        model.update_hypers(params1)
        logZ1, _, _ = model.compute_emission_log_lik_exp(
            mx, vx, scale)

        params2 = copy.deepcopy(params)
        params2['R'][a] = params2['R'][a] - eps
        model.update_hypers(params2)
        logZ2, _, _ = model.compute_emission_log_lik_exp(
            mx, vx, scale)

        da = (logZ1 - logZ2) / eps / 2
        print('R a=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
              % (a, grad_params['R'][a], da, (grad_params['R'][a] - da) / da))

    eps = 1e-5
    # check grad mx
    for a in range(N):
        for b in range(Din):
            mx1 = copy.deepcopy(mx)
            mx1[a, b] = mx1[a, b] + eps
            model.update_hypers(params)
            logZ1, _, _ = model.compute_emission_log_lik_exp(
                mx1, vx, scale)

            mx2 = copy.deepcopy(mx)
            mx2[a, b] = mx2[a, b] - eps
            model.update_hypers(params)
            logZ2, _, _ = model.compute_emission_log_lik_exp(
                mx2, vx, scale)

            dab = (logZ1 - logZ2) / eps / 2
            print('mx n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (a, b, grad_input['mx'][a, b], dab, (grad_input['mx'][a, b] - dab) / dab))

    # check grad mx
    for a in range(N):
        for b in range(Din):
            vx1 = copy.deepcopy(vx)
            vx1[a, b] = vx1[a, b] + eps
            model.update_hypers(params)
            logZ1, _, _ = model.compute_emission_log_lik_exp(
                mx, vx1, scale)

            vx2 = copy.deepcopy(vx)
            vx2[a, b] = vx2[a, b] - eps
            model.update_hypers(params)
            logZ2, _, _ = model.compute_emission_log_lik_exp(
                mx, vx2, scale)

            dab = (logZ1 - logZ2) / eps / 2
            print('vx n=%d, d=%d, computed=%.5f, numerical=%.5f, diff=%.5f'
                  % (a, b, grad_input['vx'][a, b], dab, (grad_input['vx'][a, b] - dab) / dab))


def test_gauss_emis_limit():

    # generate some datapoints for testing
    N = 5
    alpha = 0.00001
    Dout = 3
    Din = 2
    y_train = np.random.randn(N, Dout)
    # params  tied
    model = lik.Gauss_Emis(y_train, Dout, Din)

    params = {'C': np.random.randn(Dout, Din),
              'R': np.random.randn(Dout)}
    mx = np.random.randn(N, Din)
    vx = np.random.rand(N, Din)
    scale = 1

    model.update_hypers(params)
    logZ1, _, _ = model.compute_emission_tilted(
        mx, vx, alpha, 1 / alpha)

    logZ2, _, _ = model.compute_emission_log_lik_exp(
        mx, vx, scale)

    print logZ1, logZ2, logZ1 - logZ2



if __name__ == '__main__':
    for i in range(10):
        test_gauss_emis_log_tilted()
        test_gauss_emis_log_lik()
    test_gauss_emis_limit()