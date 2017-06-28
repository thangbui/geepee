print "importing stuff..."
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pdb
import matplotlib.pylab as plt
from scipy import special
import time

from .datautils import step, spiral
from .context import pep, aep

def run_regression_1D_pep_training(stoc=False):
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
    alpha = 0.1
    model_pep = pep.SGPR_rank_one(X, Y, M, lik='Gaussian')
    if stoc:
        mb_size = M
        fname = '/tmp/gpr_pep_reg_stoc.pdf'
        adam_lr = 0.005
    else:
        mb_size = N
        fname = '/tmp/gpr_pep_reg.pdf'
        adam_lr = 0.05
    model_pep.optimise(method='adam', mb_size=mb_size, adam_lr=adam_lr, alpha=alpha, maxiter=2000)
    plot(model_pep)
    plt.savefig(fname)


def run_regression_1D_pep_inference():
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
    print "create aep model and optimize ..."
    M = 20
    alpha = 0.5
    model_aep = aep.SGPR(X, Y, M, lik='Gaussian')
    model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=2000)
    # plot(model_aep)
    # plt.show()
    # plt.savefig('/tmp/gpr_aep_reg.pdf')

    start_time = time.time()
    model = pep.SGPR_rank_one(X, Y, M, lik='Gaussian')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers())
    model.run_pep(np.arange(N), 10, alpha=alpha, parallel=False, compute_energy=True)
    end_time = time.time()
    print "sequential updates: %.4f" % (end_time - start_time)
    # plot(model)
    # plt.savefig('/tmp/gpr_pep_reg_seq.pdf')


    start_time = time.time()
    model = pep.SGPR_rank_one(X, Y, M, lik='Gaussian')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers(Y))
    model.run_pep(np.arange(N), 10, alpha=alpha, parallel=True, compute_energy=False)
    end_time = time.time()
    print "parallel updates: %.4f" % (end_time - start_time)
    # plot(model)
    # plt.savefig('/tmp/gpr_pep_reg_par.pdf')

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
    alpha = 0.1
    model_aep = aep.SGPR(X, Y, M, lik='Gaussian')
    model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=2000)
    plot(model_aep)
    plt.savefig('/tmp/gpr_aep_reg.pdf')

    start_time = time.time()
    model = pep.SGPR(X, Y, M, lik='Gaussian')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers())
    model.inference(alpha=alpha, no_epochs=10)
    end_time = time.time()
    print "sequential updates: %.4f" % (end_time - start_time)
    plot(model)
    plt.savefig('/tmp/gpr_pep_reg_seq.pdf')


    start_time = time.time()
    model = pep.SGPR(X, Y, M, lik='Gaussian')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers())
    model.inference(alpha=alpha, no_epochs=10, parallel=True)
    end_time = time.time()
    print "parallel updates: %.4f" % (end_time - start_time)
    plot(model)
    plt.savefig('/tmp/gpr_pep_reg_par.pdf')

    # plt.show()


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
        plt.contour(xx, yy, mf.reshape(*xx.shape),
                    [0], colors='k', linewidths=1.8, zorder=100)

    Xtrain = np.loadtxt('./examples/data/banana_X_train.txt', delimiter=',')
    Ytrain = np.loadtxt('./examples/data/banana_Y_train.txt',
                        delimiter=',').reshape(-1, 1)
    Ytrain[np.where(Ytrain == 0)[0]] = -1
    M = 50
    alpha = 0.2
    model_aep = aep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=200)
    # plot(model_aep)
    # plt.savefig('/tmp/gpr_aep_cla.pdf')

    start_time = time.time()
    model = pep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers())
    model.inference(alpha=alpha, no_epochs=10)
    end_time = time.time()
    print "sequential updates: %.4f" % (end_time - start_time)
    # plot(model)
    # plt.savefig('/tmp/gpr_pep_cla_seq.pdf')


    start_time = time.time()
    model = pep.SGPR(Xtrain, Ytrain, M, lik='Probit')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers())
    model.inference(alpha=alpha, no_epochs=10, parallel=True)
    end_time = time.time()
    print "parallel updates: %.4f" % (end_time - start_time)
    # plot(model)
    # plt.savefig('/tmp/gpr_pep_cla_par.pdf')
    # plt.show()

def run_banana_pep_training(stoc=False):

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
        plt.contour(xx, yy, mf.reshape(*xx.shape),
                    [0], colors='k', linewidths=1.8, zorder=100)

    Xtrain = np.loadtxt('./examples/data/banana_X_train.txt', delimiter=',')
    Ytrain = np.loadtxt('./examples/data/banana_Y_train.txt',
                        delimiter=',').reshape(-1, 1)
    Ytrain[np.where(Ytrain == 0)[0]] = -1
    M = 50
    alpha = 0.01
    model_pep = pep.SGPR_rank_one(Xtrain, Ytrain, M, lik='Probit')
    if stoc:
        mb_size = M
        fname = '/tmp/gpr_pep_cla_stoc.pdf'
        adam_lr = 0.005
    else:
        mb_size = N
        fname = '/tmp/gpr_pep_cla.pdf'
        adam_lr = 0.05
    model_pep.optimise(method='adam', mb_size=mb_size, adam_lr=adam_lr, alpha=alpha, maxiter=2000)
    plot(model_pep)
    plt.savefig(fname)



def run_step_1D():
    np.random.seed(42)

    print "create dataset ..."
    N = 200
    X = np.random.rand(N, 1) * 3 - 1.5
    Y = step(X)
    # plt.plot(X, Y, 'kx', mew=2)

    def plot(m):
        xx = np.linspace(-3, 3, 100)[:, None]
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

        no_samples = 20
        xx = np.linspace(-3, 3, 500)[:, None]
        f_samples = m.sample_f(xx, no_samples)
        for i in range(no_samples):
            plt.plot(xx, f_samples[:, :, i], linewidth=0.5, alpha=0.5)

        plt.xlim(-3, 3)

    # inference
    print "create model and optimize ..."
    M = 20
    alpha = 0.01
    model_aep = aep.SGPR(X, Y, M, lik='Gaussian')
    model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=2000)
    plot(model_aep)
    plt.savefig('/tmp/gpr_aep_step.pdf')

    start_time = time.time()
    model = pep.SGPR(X, Y, M, lik='Gaussian')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers())
    model.inference(alpha=alpha, no_epochs=10)
    end_time = time.time()
    print "sequential updates: %.4f" % (end_time - start_time)
    plot(model)
    plt.savefig('/tmp/gpr_pep_step_seq.pdf')

    start_time = time.time()
    model = pep.SGPR(X, Y, M, lik='Gaussian')
    model.update_hypers(model_aep.get_hypers())
    # model.update_hypers(model.init_hypers())
    model.inference(alpha=alpha, no_epochs=1000, parallel=True)
    end_time = time.time()
    print "parallel updates: %.4f" % (end_time - start_time)
    plot(model)
    plt.savefig('/tmp/gpr_pep_step_par.pdf')

    # plt.show()

if __name__ == '__main__':
    # run_regression_1D()
    # run_banana()
    # run_step_1D()
    # run_regression_1D_pep_training(stoc=False)
    run_regression_1D_pep_training(stoc=True)
    # run_regression_1D_pep_inference()
    run_banana_pep_training(stoc=True)
    # run_banana_pep_training(stoc=False)
