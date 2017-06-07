print "importing stuff..."
import numpy as np
import pdb
import matplotlib.pylab as plt

from .datautils import step, spiral
from .context import vfe, aep


def plot(m, Xtrain, ytrain):
    xx = np.linspace(-0.5, 1.5, 100)[:, None]
    mean, var = m.predict_y(xx)
    mean = np.reshape(mean, (xx.shape[0], 1))
    var = np.reshape(var, (xx.shape[0], 1))
    if isinstance(m, aep.SDGPR):
        zu = m.sgp_layers[0].zu
    elif isinstance(m, vfe.SGPR_collapsed):
        zu = m.zu
    else:
        zu = m.sgp_layer.zu
    mean_u, var_u = m.predict_f(zu)
    plt.figure()
    plt.plot(Xtrain, ytrain, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    # pdb.set_trace()
    plt.fill_between(
        xx[:, 0],
        mean[:, 0] - 2 * np.sqrt(var[:, 0]),
        mean[:, 0] + 2 * np.sqrt(var[:, 0]),
        color='blue', alpha=0.2)
    plt.errorbar(zu, mean_u, yerr=2 * np.sqrt(var_u), fmt='ro')
    plt.xlim(-0.1, 1.1)

def create_dataset():
    N = 400
    X = np.random.rand(N, 1)
    Y = np.sin(12 * X) + 0.5 * np.cos(25 * X) + np.random.randn(N, 1) * 0.15
    Xtrain = X[:N/2, :]
    ytrain = Y[:N/2, :]
    Xtest = X[N/2:, :]
    ytest = Y[N/2:, :]
    return Xtrain, ytrain, Xtest, ytest

def run_regression_1D_collapsed():
    np.random.seed(42)

    print "create dataset ..."
    Xtrain, ytrain, Xtest, ytest = create_dataset()

    alphas = [0.001, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    for alpha in alphas:
        M = 20
        model = vfe.SGPR_collapsed(Xtrain, ytrain, M)
        model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=1000, disp=False)
        my, vy = model.predict_y(Xtest, alpha)
        my = np.reshape(my, ytest.shape)
        vy = np.reshape(vy, ytest.shape)
        rmse = np.sqrt(np.mean((my - ytest)**2))
        ll = np.mean(-0.5 * np.log(2 * np.pi * vy) - 0.5 * (ytest - my)**2 / vy)
        nlml, _ = model.objective_function(model.get_hypers(), alpha)
        print 'alpha=%.3f, train ml=%3f, test rmse=%.3f, ll=%.3f' % (alpha, nlml, rmse, ll)
        # plot(model, Xtrain, ytrain)
        # plt.show()

    # should produce something like this
    # alpha=0.001, train ml=-64.573021, test rmse=0.169, ll=0.348
    # alpha=0.100, train ml=-64.616618, test rmse=0.169, ll=0.348
    # alpha=0.200, train ml=-64.626655, test rmse=0.169, ll=0.348
    # alpha=0.300, train ml=-64.644053, test rmse=0.169, ll=0.348
    # alpha=0.500, train ml=-64.756588, test rmse=0.169, ll=0.348
    # alpha=0.700, train ml=-68.755871, test rmse=0.169, ll=0.350
    # alpha=0.800, train ml=-72.153441, test rmse=0.167, ll=0.349
    # alpha=1.000, train ml=-71.305002, test rmse=0.169, ll=0.303


def run_regression_1D_aep():
    np.random.seed(42)

    print "create dataset ..."
    Xtrain, ytrain, Xtest, ytest = create_dataset()

    alphas = [0.001, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    for alpha in alphas:
        M = 20
        model = aep.SGPR(Xtrain, ytrain, M)
        model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=5000, disp=False)
        my, vy = model.predict_y(Xtest)
        my = np.reshape(my, ytest.shape)
        vy = np.reshape(vy, ytest.shape)
        rmse = np.sqrt(np.mean((my - ytest)**2))
        ll = np.mean(-0.5 * np.log(2 * np.pi * vy) - 0.5 * (ytest - my)**2 / vy)
        nlml, _ = model.objective_function(model.get_hypers(), Xtrain.shape[0], alpha)
        print 'alpha=%.3f, train ml=%3f, test rmse=%.3f, ll=%.3f' % (alpha, nlml, rmse, ll)
        # plot_aep(model, Xtrain, ytrain)
        # plt.show()

    # should produce something like this
    # alpha=0.001, train ml=-34.678077, test rmse=0.168, ll=0.360
    # alpha=0.100, train ml=-37.453561, test rmse=0.168, ll=0.363
    # alpha=0.200, train ml=-32.129832, test rmse=0.168, ll=0.361
    # alpha=0.300, train ml=-48.637007, test rmse=0.167, ll=0.370
    # alpha=0.500, train ml=-22.190862, test rmse=0.172, ll=0.309
    # alpha=0.700, train ml=-50.283386, test rmse=0.167, ll=0.364
    # alpha=0.800, train ml=-63.789542, test rmse=0.166, ll=0.367
    # alpha=1.000, train ml=-97.553140, test rmse=0.167, ll=0.136


# this should be identical to the run above 
def run_regression_1D_aep_one_layer():
    np.random.seed(42)

    print "create dataset ..."
    Xtrain, ytrain, Xtest, ytest = create_dataset()

    alphas = [0.001, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    for alpha in alphas:
        M = 20
        model = aep.SDGPR(Xtrain, ytrain, M, hidden_sizes=[])
        model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=5000, disp=False)
        my, vy = model.predict_y(Xtest)
        my = np.reshape(my, ytest.shape)
        vy = np.reshape(vy, ytest.shape)
        rmse = np.sqrt(np.mean((my - ytest)**2))
        ll = np.mean(-0.5 * np.log(2 * np.pi * vy) - 0.5 * (ytest - my)**2 / vy)
        nlml, _ = model.objective_function(model.get_hypers(), Xtrain.shape[0], alpha)
        print 'alpha=%.3f, train ml=%3f, test rmse=%.3f, ll=%.3f' % (alpha, nlml, rmse, ll)
        
    # should produce something like this
    # alpha=0.001, train ml=-33.531741, test rmse=0.168, ll=0.357
    # alpha=0.100, train ml=-49.729338, test rmse=0.167, ll=0.371
    # alpha=0.200, train ml=-55.470459, test rmse=0.167, ll=0.367
    # alpha=0.300, train ml=-48.519526, test rmse=0.168, ll=0.367
    # alpha=0.500, train ml=-46.729320, test rmse=0.166, ll=0.372
    # alpha=0.700, train ml=-52.346971, test rmse=0.166, ll=0.371
    # alpha=0.800, train ml=-67.410540, test rmse=0.170, ll=0.339
    # alpha=1.000, train ml=-97.784332, test rmse=0.167, ll=0.124


def run_regression_1D_aep_two_layers():
    np.random.seed(42)

    print "create dataset ..."
    Xtrain, ytrain, Xtest, ytest = create_dataset()

    alpha = 1 # other alpha is not valid here
    M = 20
    model = aep.SDGPR(Xtrain, ytrain, M, hidden_sizes=[2])
    model.optimise(method='L-BFGS-B', alpha=1, maxiter=5000, disp=False)
    my, vy = model.predict_y(Xtest)
    my = np.reshape(my, ytest.shape)
    vy = np.reshape(vy, ytest.shape)
    rmse = np.sqrt(np.mean((my - ytest)**2))
    ll = np.mean(-0.5 * np.log(2 * np.pi * vy) - 0.5 * (ytest - my)**2 / vy)
    nlml, _ = model.objective_function(model.get_hypers(), Xtrain.shape[0], alpha)
    print 'alpha=%.3f, train ml=%3f, test rmse=%.3f, ll=%.3f' % (alpha, nlml, rmse, ll)
    # plot(model, Xtrain, ytrain)
    # plt.show()

    # should produce something like this
    # alpha=1.000, train ml=-51.385404, test rmse=0.168, ll=0.311


def run_regression_1D_aep_two_layers_stoc():
    np.random.seed(42)

    print "create dataset ..."
    Xtrain, ytrain, Xtest, ytest = create_dataset()

    alpha = 1 # other alpha is not valid here
    M = 20
    model = aep.SDGPR(Xtrain, ytrain, M, hidden_sizes=[2])
    model.optimise(method='adam', alpha=1, maxiter=5000, disp=False)
    my, vy = model.predict_y(Xtest)
    my = np.reshape(my, ytest.shape)
    vy = np.reshape(vy, ytest.shape)
    rmse = np.sqrt(np.mean((my - ytest)**2))
    ll = np.mean(-0.5 * np.log(2 * np.pi * vy) - 0.5 * (ytest - my)**2 / vy)
    nlml, _ = model.objective_function(model.get_hypers(), Xtrain.shape[0], alpha)
    print 'alpha=%.3f, train ml=%3f, test rmse=%.3f, ll=%.3f' % (alpha, nlml, rmse, ll)
    # plot(model, Xtrain, ytrain)
    # plt.show()

    # should produce something like this
    # alpha=1.000, train ml=-69.444086, test rmse=0.170, ll=0.318


if __name__ == '__main__':
    # running batch Power-EP for regression
    run_regression_1D_collapsed()
    # running batch approximate Power-EP for regression
    run_regression_1D_aep() 
    # running batch one layer deep GP for regression (identical to aep.GPR) 
    run_regression_1D_aep_one_layer()
    # running deep regression - network size = [D 2 1]
    run_regression_1D_aep_two_layers()
    # running deep regression with stoc. optimisation - network size = [D 2 1] 
    run_regression_1D_aep_two_layers_stoc()
