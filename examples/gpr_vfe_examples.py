print "importing stuff..."
import numpy as np
import pdb
import matplotlib.pylab as plt
from scipy import special

from .datautils import step, spiral
from .context import vfe

# import GPflow as gf


def run_regression_1D_collapsed():
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


def run_step_1D_collapsed():
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
    model = vfe.SGPR_collapsed(X, Y, M)
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


def run_regression_1D():
    np.random.seed(42)

    print "create dataset ..."
    N = 200
    X = np.random.rand(N, 1)
    Y = np.sin(12 * X) + 0.5 * np.cos(25 * X) + np.random.randn(N, 1) * 0.2
    # plt.plot(X, Y, 'kx', mew=2)

    def plot(m):
        xx = np.linspace(-0.5, 1.5, 100)[:, None]
        mean, var = m.predict_f(xx)
        zu = m.sgp_layer.zu
        mean_u, var_u = m.predict_f(zu)
        plt.figure()
        plt.plot(X, Y, 'kx', mew=2)
        plt.plot(xx, mean, 'b', lw=2)
        plt.fill_between(
            xx[:, 0],
            mean[:, 0] - 2 * np.sqrt(var[:, 0]),
            mean[:, 0] + 2 * np.sqrt(var[:, 0]),
            color='blue', alpha=0.2)
        plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
        plt.xlim(-0.1, 1.1)

    # inference
    print "create model and optimize ..."
    M = 20
    model = aep.SGPR(X, Y, M, lik='Gaussian')
    model.optimise(method='L-BFGS-B', alpha=0.1, maxiter=50000)
    plot(model)
    plt.show()


def run_banana():

    def gridParams():
        mins = [-3.25, -2.85]
        maxs = [3.65, 3.4]
        nGrid = 50
        xspaced = np.linspace(mins[0], maxs[0], nGrid)
        yspaced = np.linspace(mins[1], maxs[1], nGrid)
        xx, yy = np.meshgrid(xspaced, yspaced)
        Xplot = np.vstack((xx.flatten(), yy.flatten())).T
        return mins, maxs, xx, yy, Xplot

    def plot(m):
        col1 = '#0172B2'
        col2 = '#CC6600'
        mins, maxs, xx, yy, Xplot = gridParams()
        mf, vf = m.predict_f(Xplot)
        plt.figure()
        plt.plot(
            Xtrain[:, 0][Ytrain[:, 0] == 1],
            Xtrain[:, 1][Ytrain[:, 0] == 1],
            'o', color=col1, mew=0, alpha=0.5)
        plt.plot(
            Xtrain[:, 0][Ytrain[:, 0] == -1],
            Xtrain[:, 1][Ytrain[:, 0] == -1],
            'o', color=col2, mew=0, alpha=0.5)
        zu = m.sgp_layer.zu
        plt.plot(zu[:, 0], zu[:, 1], 'ro', mew=0, ms=4)
        plt.contour(
            xx, yy, mf.reshape(*xx.shape), [0],
            colors='k', linewidths=1.8, zorder=100)

    Xtrain = np.loadtxt(
        './examples/data/banana_X_train.txt', delimiter=',')
    Ytrain = np.loadtxt(
        './examples/data/banana_Y_train.txt', delimiter=',').reshape(-1, 1)
    Ytrain[np.where(Ytrain == 0)[0]] = -1
    M = 50
    model = aep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model.optimise(method='L-BFGS-B', alpha=0.01, maxiter=2000)
    plot(model)
    model = aep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model.optimise(method='L-BFGS-B', alpha=0.2, maxiter=2000)
    plot(model)
    model = aep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model.optimise(method='L-BFGS-B', alpha=0.7, maxiter=2000)
    plot(model)
    plt.show()


def run_regression_1D_stoc():
    np.random.seed(42)

    print "create dataset ..."
    N = 200
    X = np.random.rand(N, 1)
    Y = np.sin(12 * X) + 0.5 * np.cos(25 * X) + np.random.randn(N, 1) * 0.2
    # plt.plot(X, Y, 'kx', mew=2)

    def plot(m):
        xx = np.linspace(-0.5, 1.5, 100)[:, None]
        mean, var = m.predict_f(xx)
        zu = m.sgp_layer.zu
        mean_u, var_u = m.predict_f(zu)
        plt.figure()
        plt.plot(X, Y, 'kx', mew=2)
        plt.plot(xx, mean, 'b', lw=2)
        plt.fill_between(
            xx[:, 0],
            mean[:, 0] - 2 * np.sqrt(var[:, 0]),
            mean[:, 0] + 2 * np.sqrt(var[:, 0]),
            color='blue', alpha=0.2)
        plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
        plt.xlim(-0.1, 1.1)

    # inference
    print "create model and optimize ..."
    M = 20
    model = aep.SGPR(X, Y, M, lik='Gaussian')
    model.optimise(method='adam', alpha=0.1,
                   maxiter=100000, mb_size=M, adam_lr=0.001)
    plot(model)
    plt.show()
    plt.savefig('/tmp/aep_gpr_1D_stoc.pdf')


def run_banana_stoc():

    def gridParams():
        mins = [-3.25, -2.85]
        maxs = [3.65, 3.4]
        nGrid = 50
        xspaced = np.linspace(mins[0], maxs[0], nGrid)
        yspaced = np.linspace(mins[1], maxs[1], nGrid)
        xx, yy = np.meshgrid(xspaced, yspaced)
        Xplot = np.vstack((xx.flatten(), yy.flatten())).T
        return mins, maxs, xx, yy, Xplot

    def plot(m):
        col1 = '#0172B2'
        col2 = '#CC6600'
        mins, maxs, xx, yy, Xplot = gridParams()
        mf, vf = m.predict_f(Xplot)
        plt.figure()
        plt.plot(
            Xtrain[:, 0][Ytrain[:, 0] == 1],
            Xtrain[:, 1][Ytrain[:, 0] == 1],
            'o', color=col1, mew=0, alpha=0.5)
        plt.plot(
            Xtrain[:, 0][Ytrain[:, 0] == -1],
            Xtrain[:, 1][Ytrain[:, 0] == -1],
            'o', color=col2, mew=0, alpha=0.5)
        zu = m.sgp_layer.zu
        plt.plot(zu[:, 0], zu[:, 1], 'ro', mew=0, ms=4)
        plt.contour(
            xx, yy, mf.reshape(*xx.shape), [0],
            colors='k', linewidths=1.8, zorder=100)

    Xtrain = np.loadtxt(
        './examples/data/banana_X_train.txt', delimiter=',')
    Ytrain = np.loadtxt(
        './examples/data/banana_Y_train.txt', delimiter=',').reshape(-1, 1)
    Ytrain[np.where(Ytrain == 0)[0]] = -1
    M = 30
    model = aep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model.optimise(method='adam', alpha=0.5,
                   maxiter=100000, mb_size=M, adam_lr=0.001)
    plot(model)
    plt.show()
    plt.savefig('/tmp/aep_gpc_banana_stoc.pdf')

if __name__ == '__main__':
    run_regression_1D()
    run_step_1D()
