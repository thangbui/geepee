
import pprint
import numpy as np
import copy
import pdb

from .context import PROP_MC, PROP_MM

pp = pprint.PrettyPrinter(indent=4)


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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_message(message_str, computed, numerical):
    diff = (computed - numerical) / numerical
    if np.abs(diff) < 1e-4 or ((np.isinf(diff) or np.isnan(diff) or (np.abs(diff) <= 1.1 and np.abs(diff) > 1e-4)) and np.abs(numerical) < 1e-4):
        pass
    else:
        print(bcolors.FAIL + '%s, computed=%.5f, numerical=%.5f, rel. diff=%.5f' % (message_str, computed, numerical, diff) + bcolors.ENDC)

def check_grad(params, model, stochastic=False, alpha=0.5, stoc_seed=42, prop_mode=PROP_MM):
    if stochastic:
        if hasattr(model, 'M'):
            N = model.M
        else:
            N = int(np.ceil(model.N/3.0))
    else:
        N = model.N
    eps = 1e-5
    fixed_seed = stochastic or prop_mode == PROP_MC
    if fixed_seed:
        np.random.seed(stoc_seed)
    logZ, grad_all = model.objective_function(params, N, alpha=alpha, prop_mode=prop_mode)
    for name in params.keys():
        if hasattr(params[name], "__len__"):
            pshape = params[name].shape
            if len(pshape) == 2:
                for i in range(pshape[0]):
                    for j in range(pshape[1]):
                        params_plus = copy.deepcopy(params)
                        params_plus[name][i, j] += eps
                        if fixed_seed:
                            np.random.seed(stoc_seed)
                        val_plus, _ = model.objective_function(params_plus, N, alpha=alpha, prop_mode=prop_mode)
                        params_minus = copy.deepcopy(params)
                        params_minus[name][i, j] -= eps
                        if fixed_seed:
                            np.random.seed(stoc_seed)
                        val_minus, _ = model.objective_function(params_minus, N, alpha=alpha, prop_mode=prop_mode)

                        dij = (val_plus - val_minus) / eps / 2
                        print_message('%s i=%d/%d, j=%d/%d' % (name, i, pshape[0], j, pshape[1]), grad_all[name][i, j], dij)
            elif len(pshape) == 1:
                for i in range(pshape[0]):
                    params_plus = copy.deepcopy(params)
                    params_plus[name][i] += eps
                    if fixed_seed:
                        np.random.seed(stoc_seed)
                    val_plus, _ = model.objective_function(params_plus, N, alpha=alpha, prop_mode=prop_mode)
                    params_minus = copy.deepcopy(params)
                    params_minus[name][i] -= eps
                    if fixed_seed:
                        np.random.seed(stoc_seed)
                    val_minus, _ = model.objective_function(params_minus, N, alpha=alpha, prop_mode=prop_mode)

                    di = (val_plus - val_minus) / eps / 2
                    if hasattr(grad_all[name], '__len__'):
                        g = grad_all[name][i]
                    else:
                        g = grad_all[name]
                    print_message('%s i=%d/%d' % (name, i, pshape[0]), g, di)
            else: # len(pshape) = 0
                params_plus = copy.deepcopy(params)
                params_plus[name] += eps
                if fixed_seed:
                    np.random.seed(stoc_seed)
                val_plus, _ = model.objective_function(params_plus, N, alpha=alpha, prop_mode=prop_mode)
                params_minus = copy.deepcopy(params)
                params_minus[name] -= eps
                if fixed_seed:
                    np.random.seed(stoc_seed)
                val_minus, _ = model.objective_function(params_minus, N, alpha=alpha, prop_mode=prop_mode)

                di = (val_plus - val_minus) / eps / 2
                g = grad_all[name]
                print_message('%s' % name, g, di)
        else:
            params_plus = copy.deepcopy(params)
            params_plus[name] += eps
            if fixed_seed:
                np.random.seed(stoc_seed)
            val_plus, _ = model.objective_function(params_plus, N, alpha=alpha, prop_mode=prop_mode)
            params_minus = copy.deepcopy(params)
            params_minus[name] -= eps
            if fixed_seed:
                np.random.seed(stoc_seed)
            val_minus, _ = model.objective_function(params_minus, N, alpha=alpha, prop_mode=prop_mode)

            d = (val_plus - val_minus) / eps / 2
            print_message('%s' % (name), grad_all[name], d)
