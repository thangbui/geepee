print "importing stuff..."
import numpy as np
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from scipy import special

from .datautils import step, spiral
from .context import aep

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


def run_step_1D():
    np.random.seed(42)

    print "create dataset ..."
    N = 200
    X = np.random.rand(N, 1) * 3 - 1.5
    Y = step(X)
    # plt.plot(X, Y, 'kx', mew=2)

    def plot(m):
        xx = np.linspace(-3, 3, 100)[:, None]
        mean, var = m.predict_y(xx)
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

        no_samples = 20
        xx = np.linspace(-3, 3, 500)[:, None]
        f_samples = m.sample_f(xx, no_samples)
        for i in range(no_samples):
            plt.plot(xx, f_samples[:, :, i], linewidth=0.5, alpha=0.5)

        plt.xlim(-3, 3)

    # inference
    print "create model and optimize ..."
    M = 20
    model = aep.SGPR(X, Y, M, lik='Gaussian')
    model.optimise(method='L-BFGS-B', alpha=0.9, maxiter=2000)
    plot(model)
    plt.savefig('/tmp/aep_gpr_step.pdf')
    # plt.show()


def run_spiral():
    np.random.seed(42)

    def gridParams():
        mins = [-1.2, -1.2]
        maxs = [1.2, 1.2]
        nGrid = 80
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
        plt.contour(xx, yy, mf.reshape(*xx.shape), [0],
                    colors='k', linewidths=1.8, zorder=100)

    N = 100
    M = 50
    Xtrain, Ytrain = spiral(N)
    Xtrain /= 6
    model = aep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model.set_fixed_params(['sf'])
    model.optimise(method='L-BFGS-B', alpha=1, maxiter=2000)
    plot(model)
    plt.show()


def run_boston():
    np.random.seed(100)
    # We load the dataset
    # datapath = '/scratch/tdb40/datasets/reg/bostonHousing/data/'
    datapath = '/tmp/bostonHousing/data/'
    datafile = datapath + 'data.txt'
    data = np.loadtxt(datafile)

    # We obtain the features and the targets
    xindexfile = datapath + 'index_features.txt'
    yindexfile = datapath + 'index_target.txt'
    xindices = np.loadtxt(xindexfile, dtype=np.int)
    yindex = np.loadtxt(yindexfile, dtype=np.int)
    X = data[:, xindices]
    y = data[:, yindex]
    y = y.reshape([y.shape[0], 1])
    train_ind_file = datapath + 'index_train_0.txt'
    test_ind_file = datapath + 'index_test_0.txt'
    index_train = np.loadtxt(train_ind_file, dtype=np.int)
    index_test = np.loadtxt(test_ind_file, dtype=np.int)
    X_train = X[index_train, :]
    y_train = y[index_train, :]
    X_test = X[index_test, :]
    y_test = y[index_test, :]

    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train_normalized = (y_train - mean_y_train) / std_y_train

    M = 50
    alpha = 0.5
    model = aep.SGPR(X_train, y_train_normalized, M, lik='Gaussian')
    model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=2000)

    my, vy = model.predict_y(X_test)
    my = std_y_train * my + mean_y_train
    vy = std_y_train**2 * vy
    # We compute the test RMSE
    test_rmse = np.sqrt(np.mean((y_test - my)**2))
    print 'RMSE %.3f' % test_rmse

    # We compute the test log-likelihood
    test_nll = np.mean(-0.5 * np.log(2 * np.pi * vy) -
                       0.5 * (y_test - my)**2 / vy)
    print 'MLL %.3f' % test_nll


if __name__ == '__main__':
    # run_regression_1D()
    # run_banana()
    run_step_1D()
    # run_spiral()
    # run_boston()

    # run_regression_1D_stoc()
    # run_banana_stoc()
