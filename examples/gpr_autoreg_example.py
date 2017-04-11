print "importing stuff..."
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pdb
import matplotlib.pylab as plt
from scipy import special

from .context import vfe, compute_kernel, compute_psi_weave
jitter = 1e-5


class AutoSGPR(object):

    def __init__(self, X_train, Y_train, M):
        self.N_train = Y_train.shape[0]
        self.Dout = Y_train.shape[1]
        self.Din = Y_train.shape[1]
        self.M = M
        models = []
        for i in range(self.Dout):
            if i == 0:
                X_train_i = X_train
            else:
                X_train_i = np.hstack(
                    (X_train, Y_train[:, :i].reshape((self.N_train, i))))
            y_train_i = Y_train[:, [i]]
            model_i = vfe.SGPR(X_train_i, y_train_i, M)
            models.append(model_i)
        self.models = models

    def train(self, alpha=0.5):
        for i, model in enumerate(self.models):
            print "Training model %d" % i
            # model.set_fixed_params(['sn'])
            np.random.seed(i)
            model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=5000)

    def predict_f(self, X_test, alpha, use_mean_only=True):
        N_test = X_test.shape[0]
        mf = np.zeros((N_test, self.Dout))
        vf = np.zeros((N_test, self.Dout))
        for i, model in enumerate(self.models):
            if i == 0:
                res = model.predict_f(X_test, alpha=alpha)
                mf[:, i], vf[:, i] = res[0][:, 0], res[1]
            else:
                if use_mean_only:
                    X_test_i = np.hstack(
                        (X_test, mf[:, :i].reshape((N_test, i))))
                    res = model.predict_f(X_test_i, alpha=alpha)
                    mf[:, i], vf[:, i] = res[0][:, 0], res[1]
                else:
                    mu, Su = model.predict_f(
                        model.zu, alpha=alpha, marginal=False)
                    m_X_test_i = np.hstack(
                        (X_test, mf[:, :i].reshape((N_test, i))))
                    v_X_test_i = np.hstack(
                        (np.zeros_like(X_test), vf[:, :i].reshape((N_test, i))))
                    res = self.forward_prop_random_thru_post_mm(
                        model, m_X_test_i, v_X_test_i, mu, Su)
                    mf[:, i], vf[:, i] = res[0][:, 0], res[1][:, 0]

        return mf, vf

    def forward_prop_deterministic_thru_post(self, model, x, mu, Su):
        Kuu_noiseless = compute_kernel(
            2 * model.ls, 2 * model.sf, model.zu, model.zu)
        Kuu = Kuu_noiseless + np.diag(jitter * np.ones((self.M, )))
        # TODO: probably not the best way to use inv
        Kuuinv = np.linalg.inv(Kuu)
        A = np.dot(Kuuinv, mu)
        Smm = Su + np.outer(mu, mu)
        B_det = np.dot(Kuuinv, np.dot(Su, Kuuinv)) - Kuuinv
        psi0 = np.exp(2 * model.sf)
        psi1 = compute_kernel(2 * model.ls, 2 * model.sf, x, model.zu)
        mout = np.einsum('nm,md->nd', psi1, A)
        Bpsi2 = np.einsum('ab,na,nb->n', B_det, psi1, psi1)[:, np.newaxis]
        vout = psi0 + Bpsi2
        return mout, vout

    def forward_prop_random_thru_post_mm(self, model, mx, vx, mu, Su):
        Kuu_noiseless = compute_kernel(
            2 * model.ls, 2 * model.sf, model.zu, model.zu)
        Kuu = Kuu_noiseless + np.diag(jitter * np.ones((self.M, )))
        # TODO: remove inv
        Kuuinv = np.linalg.inv(Kuu)
        A = np.dot(Kuuinv, mu)
        Smm = Su + np.outer(mu, mu)
        B_sto = np.dot(Kuuinv, np.dot(Smm, Kuuinv)) - Kuuinv
        psi0 = np.exp(2.0 * model.sf)
        psi1, psi2 = compute_psi_weave(
            2 * model.ls, 2 * model.sf, mx, vx, model.zu)
        mout = np.einsum('nm,md->nd', psi1, A)
        Bpsi2 = np.einsum('ab,nab->n', B_sto, psi2)[:, np.newaxis]
        vout = psi0 + Bpsi2 - mout**2
        return mout, vout


class IndepSGPR(object):

    def __init__(self, X_train, Y_train, M):
        self.N_train = Y_train.shape[0]
        self.Dout = Y_train.shape[1]
        self.Din = Y_train.shape[1]
        self.M = M
        models = []
        for i in range(self.Dout):
            X_train_i = X_train
            y_train_i = Y_train[:, [i]]
            model_i = vfe.SGPR(X_train_i, y_train_i, M)
            models.append(model_i)
        self.models = models

    def train(self, alpha=0.5):
        for i, model in enumerate(self.models):
            print "Training model %d" % i
            # model.set_fixed_params(['sn'])
            np.random.seed(i)
            model.optimise(method='L-BFGS-B', alpha=alpha, maxiter=5000)

    def predict_f(self, X_test, alpha):
        N_test = X_test.shape[0]
        mf = np.zeros((N_test, self.Dout))
        vf = np.zeros((N_test, self.Dout))
        for i, model in enumerate(self.models):
            res = model.predict_f(X_test, alpha=alpha)
            mf[:, i], vf[:, i] = res[0][:, 0], res[1]
        return mf, vf


def run_regression_1D():
    np.random.seed(42)

    print "create dataset ..."
    N = 50
    rng = np.random.RandomState(42)
    X = np.sort(2 * rng.rand(N, 1) - 1, axis=0)
    Y = np.array([np.pi * np.sin(10 * X).ravel(),
                  np.pi * np.cos(10 * X).ravel()]).T
    Y += (0.5 - rng.rand(*Y.shape))
    Y = Y / np.std(Y, axis=0)

    def plot(model, alpha, fname):
        xx = np.linspace(-1.2, 1.2, 200)[:, None]
        if isinstance(model, IndepSGPR):
            mf, vf = model.predict_f(xx, alpha)
        else:
            # mf, vf = model.predict_f(xx, alpha, use_mean_only=False)
            mf, vf = model.predict_f(xx, alpha, use_mean_only=True)

        colors = ['r', 'b']
        plt.figure()
        for i in range(model.Dout):
            plt.subplot(model.Dout, 1, i + 1)
            plt.plot(X, Y[:, i], 'x', color=colors[i], mew=2)
            zu = model.models[i].zu
            mean_u, var_u = model.models[i].predict_f(zu, alpha)
            plt.plot(xx, mf[:, i], '-', color=colors[i], lw=2)
            plt.fill_between(
                xx[:, 0],
                mf[:, i] - 2 * np.sqrt(vf[:, i]),
                mf[:, i] + 2 * np.sqrt(vf[:, i]),
                color=colors[i], alpha=0.3)
            # plt.errorbar(zu[:, 0], mean_u, yerr=2*np.sqrt(var_u), fmt='ro')
            plt.xlim(-1.2, 1.2)
        plt.savefig(fname)

    # inference
    print "create independent output model and optimize ..."
    M = N
    alpha = 0.01
    indep_model = IndepSGPR(X, Y, M)
    indep_model.train(alpha=alpha)
    plot(indep_model, alpha, '/tmp/reg_indep_multioutput.pdf')

    print "create correlated output model and optimize ..."
    M = N
    ar_model = AutoSGPR(X, Y, M)
    ar_model.train(alpha=alpha)
    plot(ar_model, alpha, '/tmp/reg_autoreg_multioutput.pdf')

if __name__ == '__main__':
    run_regression_1D()
