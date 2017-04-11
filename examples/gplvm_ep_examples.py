print "importing stuff..."
import numpy as np
import pdb
import matplotlib.pylab as plt
from .context import aep, ep
from scipy import special

np.random.seed(42)


def run_cluster():
    import GPy
    # create dataset
    print "creating dataset..."
    N = 50
    k1 = GPy.kern.RBF(5, variance=1, lengthscale=1. /
                      np.random.dirichlet(np.r_[10, 10, 10, 0.1, 0.1]), ARD=True)
    k2 = GPy.kern.RBF(5, variance=1, lengthscale=1. /
                      np.random.dirichlet(np.r_[10, 0.1, 10, 0.1, 10]), ARD=True)
    k3 = GPy.kern.RBF(5, variance=1, lengthscale=1. /
                      np.random.dirichlet(np.r_[0.1, 0.1, 10, 10, 10]), ARD=True)
    X = np.random.normal(0, 1, (N, 5))
    A = np.random.multivariate_normal(np.zeros(N), k1.K(X), 10).T
    B = np.random.multivariate_normal(np.zeros(N), k2.K(X), 10).T
    C = np.random.multivariate_normal(np.zeros(N), k3.K(X), 10).T

    Y = np.vstack((A, B, C))
    labels = np.hstack((np.zeros(A.shape[0]), np.ones(
        B.shape[0]), np.ones(C.shape[0]) * 2))

    # inference
    print "inference ..."
    M = 20
    D = 5
    lvm_aep = aep.SGPLVM(Y, D, M, lik='Gaussian')
    lvm_aep.optimise(method='L-BFGS-B', alpha=0.1, maxiter=2000)

    lvm = ep.SGPLVM(Y, D, M, lik='Gaussian')
    lvm.update_hypers(lvm_aep.get_hypers())
    # # a quick hack to initialise the factors
    # lvm.sgp_layer.t1 = np.tile(lvm_aep.sgp_layer.theta_2[np.newaxis, :, :] / lvm.N, [lvm.N, 1, 1])
    # lvm.sgp_layer.t2 = np.tile(lvm_aep.sgp_layer.theta_1[np.newaxis, :, :, :] / lvm.N, [lvm.N, 1, 1, 1])
    # lvm.sgp_layer.update_posterior()
    # lvm.tx1 = lvm_aep.factor_x1
    # lvm.tx2 = lvm_aep.factor_x2
    lvm.inference(alpha=0.1, no_epochs=10, parallel=True, decay=0.5)

    ls = np.exp(lvm.sgp_layer.ls)
    print ls
    inds = np.argsort(ls)
    plt.figure()
    mx, vx = lvm.get_posterior_x()
    plt.scatter(mx[:, inds[0]], mx[:, inds[1]], c=labels)
    zu = lvm.sgp_layer.zu
    plt.plot(zu[:, inds[0]], zu[:, inds[1]], 'ko')
    plt.show()


def run_oil():
    data_path = './tmp/data/'

    def oil(data_set='oil'):
        """The three phase oil data from Bishop and James (1993)."""
        oil_train_file = os.path.join(data_path, data_set, 'DataTrn.txt')
        oil_trainlbls_file = os.path.join(
            data_path, data_set, 'DataTrnLbls.txt')
        oil_test_file = os.path.join(data_path, data_set, 'DataTst.txt')
        oil_testlbls_file = os.path.join(
            data_path, data_set, 'DataTstLbls.txt')
        oil_valid_file = os.path.join(data_path, data_set, 'DataVdn.txt')
        oil_validlbls_file = os.path.join(
            data_path, data_set, 'DataVdnLbls.txt')
        fid = open(oil_train_file)
        X = np.fromfile(fid, sep='\t').reshape((-1, 12))
        fid.close()
        fid = open(oil_test_file)
        Xtest = np.fromfile(fid, sep='\t').reshape((-1, 12))
        fid.close()
        fid = open(oil_valid_file)
        Xvalid = np.fromfile(fid, sep='\t').reshape((-1, 12))
        fid.close()
        fid = open(oil_trainlbls_file)
        Y = np.fromfile(fid, sep='\t').reshape((-1, 3)) * 2. - 1.
        fid.close()
        fid = open(oil_testlbls_file)
        Ytest = np.fromfile(fid, sep='\t').reshape((-1, 3)) * 2. - 1.
        fid.close()
        fid = open(oil_validlbls_file)
        Yvalid = np.fromfile(fid, sep='\t').reshape((-1, 3)) * 2. - 1.
        fid.close()
        return {'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'Xtest': Xtest, 'Xvalid': Xvalid, 'Yvalid': Yvalid}

    def oil_100(data_set='oil'):
        data = oil()
        indices = np.random.permutation(1000)
        indices = indices[0:100]
        X = data['X'][indices, :]
        Y = data['Y'][indices, :]
        return {'X': X, 'Y': Y, 'info': "Subsample of the full oil data extracting 100 values randomly without replacement"}

    # create dataset
    print "loading dataset..."
    # data = oil_100()
    data = oil()
    Y = data['X']
    # Y_mean = np.mean(Y, axis=0)
    # Y_std = np.std(Y, axis=0)
    # Y = (Y - Y_mean) / Y_std
    labels = data['Y'].argmax(axis=1)

    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))

    # inference
    print "inference ..."
    M = 20
    D = 5
    lvm = aep.SGPLVM(Y, D, M, lik='Gaussian')
    # lvm.set_fixed_params('sn')
    lvm.optimise(method='L-BFGS-B', alpha=0.3, maxiter=3000)

    # np.random.seed(0)
    # # lvm.set_fixed_params('sn')
    # lvm.optimise(method='adam', alpha=0.2, adam_lr=0.05, maxiter=200)

    ls = np.exp(lvm.sgp_layer.ls)
    print ls
    inds = np.argsort(ls)
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    plt.figure()
    mx, vx = lvm.get_posterior_x()
    plt.scatter(mx[:, inds[0]], mx[:, inds[1]], c=labels)
    zu = lvm.sgp_layer.zu
    plt.plot(zu[:, inds[0]], zu[:, inds[1]], 'ko')
    plt.show()


def run_pinwheel():
    def make_pinwheel(radial_std, tangential_std, num_classes, num_per_class, rate,
                      rs=np.random.RandomState(0)):
        """Based on code by Ryan P. Adams."""
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rs.randn(num_classes * num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles),
                              np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return np.einsum('ti,tij->tj', features, rotations)

    # create dataset
    print "creating dataset..."
    Y = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
                      num_per_class=50, rate=0.4)

    # inference
    print "inference ..."
    M = 20
    D = 2
    lvm_aep = aep.SGPLVM(Y, D, M, lik='Gaussian')
    lvm_aep.optimise(method='L-BFGS-B', alpha=0.2)

    lvm = ep.SGPLVM(Y, D, M, lik='Gaussian')
    lvm.update_hypers(lvm_aep.get_hypers())
    # # a quick hack to initialise the factors
    # lvm.sgp_layer.t1 = np.tile(lvm_aep.sgp_layer.theta_2[np.newaxis, :, :] / lvm.N, [lvm.N, 1, 1])
    # lvm.sgp_layer.t2 = np.tile(lvm_aep.sgp_layer.theta_1[np.newaxis, :, :, :] / lvm.N, [lvm.N, 1, 1, 1])
    # lvm.sgp_layer.update_posterior()
    # lvm.tx1 = lvm_aep.factor_x1
    # lvm.tx2 = lvm_aep.factor_x2
    lvm.inference(alpha=0.1, no_epochs=100, parallel=True, decay=0.5)

    mx, vx = lvm.get_posterior_x()

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(Y[:, 0], Y[:, 1], 'bx')
    ax = fig.add_subplot(122)
    ax.errorbar(mx[:, 0], mx[:, 1], xerr=np.sqrt(
        vx[:, 0]), yerr=np.sqrt(vx[:, 1]), fmt='xk')
    plt.show()


def run_semicircle():
    # create dataset
    print "creating dataset..."
    N = 20
    cos_val = [0.97, 0.95, 0.94, 0.89, 0.8,
               0.88, 0.92, 0.96, 0.7, 0.65,
               0.3, 0.25, 0.1, -0.25, -0.3,
               -0.6, -0.67, -0.75, -0.97, -0.98]
    cos_val = np.array(cos_val).reshape((N, 1))
    # cos_val = 2*np.random.rand(N, 1) - 1
    angles = np.arccos(cos_val)
    sin_val = np.sin(angles)
    Y = np.hstack((sin_val, cos_val))
    Y += 0.05 * np.random.randn(Y.shape[0], Y.shape[1])

    # inference
    print "inference ..."
    M = 10
    D = 1
    alpha = 0.2
    lvm_aep = aep.SGPLVM(Y, D, M, lik='Gaussian')
    lvm_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=2000)

    plt.figure()
    plt.plot(Y[:, 0], Y[:, 1], 'sb')

    mx, vx = lvm_aep.get_posterior_x()
    for i in range(mx.shape[0]):
        mxi = mx[i, :]
        vxi = vx[i, :]
        mxi1 = mxi + np.sqrt(vxi)
        mxi2 = mxi - np.sqrt(vxi)
        mxis = np.vstack([mxi.reshape((1, D)),
                          mxi1.reshape((1, D)),
                          mxi2.reshape((1, D))])
        myis, vyis = lvm_aep.predict_f(mxis)

        plt.errorbar(myis[:, 0], myis[:, 1],
                     xerr=np.sqrt(vyis[:, 0]), yerr=np.sqrt(vyis[:, 1]), fmt='.k')

    lvm = ep.SGPLVM(Y, D, M, lik='Gaussian')
    hypers = lvm_aep.get_hypers()
    # hypers['sn'] = np.log(0.00001)
    lvm.update_hypers(hypers)

    # # a quick hack to initialise the factors
    # lvm.sgp_layer.t1 = np.tile(lvm_aep.sgp_layer.theta_2[np.newaxis, :, :] / lvm.N, [lvm.N, 1, 1])
    # lvm.sgp_layer.t2 = np.tile(lvm_aep.sgp_layer.theta_1[np.newaxis, :, :, :] / lvm.N, [lvm.N, 1, 1, 1])
    # lvm.sgp_layer.update_posterior()
    # lvm.tx1 = lvm_aep.factor_x1
    # lvm.tx2 = lvm_aep.factor_x2

    lvm.inference(alpha=alpha, no_epochs=200, parallel=True, decay=0.5)

    plt.figure()
    plt.plot(Y[:, 0], Y[:, 1], 'sb')

    mx, vx = lvm.get_posterior_x()
    for i in range(mx.shape[0]):
        mxi = mx[i, :]
        vxi = vx[i, :]
        mxi1 = mxi + np.sqrt(vxi)
        mxi2 = mxi - np.sqrt(vxi)
        mxis = np.vstack([mxi.reshape((1, D)),
                          mxi1.reshape((1, D)),
                          mxi2.reshape((1, D))])
        myis, vyis = lvm.predict_f(mxis)

        plt.errorbar(myis[:, 0], myis[:, 1],
                     xerr=np.sqrt(vyis[:, 0]), yerr=np.sqrt(vyis[:, 1]), fmt='.k')

    plt.show()


def run_xor():
    from operator import xor

    from scipy import special
    # create dataset
    print "generating dataset..."

    n = 25
    Y = np.zeros((0, 3))
    for i in [0, 1]:
        for j in [0, 1]:
            a = i * np.ones((n, 1))
            b = j * np.ones((n, 1))
            c = xor(bool(i), bool(j)) * np.ones((n, 1))
            Y_ij = np.hstack((a, b, c))
            Y = np.vstack((Y, Y_ij))

    Ytrain = 2 * Y - 1

    # inference
    print "inference ..."
    M = 10
    D = 2
    alpha = 0.1
    lvm_aep = aep.SGPLVM(Ytrain, D, M, lik='Probit')
    lvm_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=2000)

    # predict given inputs
    mx, vx = lvm_aep.get_posterior_x()
    lims = [-1.5, 1.5]
    x = np.linspace(*lims, num=101)
    y = np.linspace(*lims, num=101)
    X, Y = np.meshgrid(x, y)
    X_ravel = X.ravel()
    Y_ravel = Y.ravel()
    inputs = np.vstack((X_ravel, Y_ravel)).T
    my, vy = lvm_aep.predict_f(inputs)
    t = my / np.sqrt(1 + vy)
    Z = 0.5 * (1 + special.erf(t / np.sqrt(2)))
    for d in range(3):
        plt.figure()
        plt.scatter(mx[:, 0], mx[:, 1])
        zu = lvm_aep.sgp_layer.zu
        plt.plot(zu[:, 0], zu[:, 1], 'ko')
        plt.contour(X, Y, np.log(Z[:, d] + 1e-16).reshape(X.shape))
        plt.xlim(*lims)
        plt.ylim(*lims)

    lvm = ep.SGPLVM(Ytrain, D, M, lik='Probit')
    lvm.update_hypers(lvm_aep.get_hypers())
    # # a quick hack to initialise the factors
    # lvm.sgp_layer.t1 = np.tile(lvm_aep.sgp_layer.theta_2[np.newaxis, :, :] / lvm.N, [lvm.N, 1, 1])
    # lvm.sgp_layer.t2 = np.tile(lvm_aep.sgp_layer.theta_1[np.newaxis, :, :, :] / lvm.N, [lvm.N, 1, 1, 1])
    # lvm.sgp_layer.update_posterior()
    # lvm.tx1 = lvm_aep.factor_x1
    # lvm.tx2 = lvm_aep.factor_x2
    lvm.inference(alpha=alpha, no_epochs=100, parallel=True, decay=0.5)

    # predict given inputs
    mx, vx = lvm.get_posterior_x()
    lims = [-1.5, 1.5]
    x = np.linspace(*lims, num=101)
    y = np.linspace(*lims, num=101)
    X, Y = np.meshgrid(x, y)
    X_ravel = X.ravel()
    Y_ravel = Y.ravel()
    inputs = np.vstack((X_ravel, Y_ravel)).T
    my, vy = lvm.predict_f(inputs)
    t = my / np.sqrt(1 + vy)
    Z = 0.5 * (1 + special.erf(t / np.sqrt(2)))
    for d in range(3):
        plt.figure()
        plt.scatter(mx[:, 0], mx[:, 1])
        zu = lvm.sgp_layer.zu
        plt.plot(zu[:, 0], zu[:, 1], 'ko')
        plt.contour(X, Y, np.log(Z[:, d] + 1e-16).reshape(X.shape))
        plt.xlim(*lims)
        plt.ylim(*lims)

    plt.show()


if __name__ == '__main__':
    # run_cluster()
    # run_semicircle()
    # run_pinwheel()
    run_xor()
