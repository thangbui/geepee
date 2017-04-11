print "importing stuff..."
import numpy as np
import pdb
import matplotlib.pylab as plt
from scipy import special

from .datautils import step, spiral
from .context import vfe

import GPflow as gf


def run_regression_1D():
    np.random.seed(42)

    print "create dataset ..."
    N = 200
    X = np.random.rand(N, 1)
    Y = np.sin(12 * X) + 0.5 * np.cos(25 * X) + np.random.randn(N, 1) * 0.2
    # plt.plot(X, Y, 'kx', mew=2)

    def plot(m):
        xx = np.linspace(-0.5, 1.5, 100)[:, None]
        mean, var = m.predict_f(xx, alpha)
        zu = m.zu
        mean_u, var_u = m.predict_f(zu)
        plt.figure()
        plt.plot(X, Y, 'kx', mew=2)
        plt.plot(xx, mean, 'b', lw=2)
        # pdb.set_trace()
        plt.fill_between(
            xx[:, 0],
            mean[:, 0] - 2 * np.sqrt(var),
            mean[:, 0] + 2 * np.sqrt(var),
            color='blue', alpha=0.2)
        plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
        plt.xlim(-0.1, 1.1)

    # inference
    print "create model and optimize ..."
    M = 20
    alpha = 0.01
    model = vfe.SGPR(X, Y, M)
    params = model.init_hypers()
    # model.update_hypers(params)
    model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=1000)
    plot(model)

    # print model.objective_function(params, alpha=1.0)[0]
    # print model.objective_function_manual(params, alpha=1.0)

    # gfm = gf.sgpr.SGPR(X, Y, gf.kernels.RBF(1), Z=params['zu'])
    # gfm.likelihood.variance = np.exp(2*params['sn'])
    # gfm.kern.variance = np.exp(2*params['sf'])
    # gfm.kern.lengthscales = np.exp(params['ls'])

    # print gfm.compute_log_likelihood()

    plt.show()


def run_step_1D():
    np.random.seed(42)

    print "create dataset ..."
    N = 200
    X = np.random.rand(N, 1) * 3 - 1.5
    Y = step(X)
    # plt.plot(X, Y, 'kx', mew=2)

    def plot(m):
        xx = np.linspace(-3, 3, 100)[:, None]
        mean, var = m.predict_f(xx, alpha)
        zu = m.zu
        mean_u, var_u = m.predict_f(zu)
        plt.figure()
        plt.plot(X, Y, 'kx', mew=2)
        plt.plot(xx, mean, 'b', lw=2)
        plt.fill_between(
            xx[:, 0],
            mean[:, 0] - 2 * np.sqrt(var),
            mean[:, 0] + 2 * np.sqrt(var),
            color='blue', alpha=0.2)
        plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')

        # no_samples = 20
        # f_samples = m.sample_f(xx, no_samples)
        # for i in range(no_samples):
        # 	plt.plot(xx, f_samples[:, :, i], linewidth=0.5, alpha=0.5)

        plt.xlim(-3, 3)

    # inference
    print "create model and optimize ..."
    M = 20
    alpha = 0.01
    model = vfe.SGPR(X, Y, M)
    # params = model.init_hypers()
    # model.update_hypers(params)
    model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=1000)
    plot(model)

    # print model.objective_function(params, alpha=1.0)

    # gfm = gf.sgpr.GPRFITC(X, Y, gf.kernels.RBF(1), Z=params['zu'])
    # gfm.likelihood.variance = np.exp(2*params['sn'])
    # gfm.kern.variance = np.exp(2*params['sf'])
    # gfm.kern.lengthscales = np.exp(params['ls'])

    # print gfm.compute_log_likelihood()

    plt.show()


if __name__ == '__main__':
    run_regression_1D()
    run_step_1D()
