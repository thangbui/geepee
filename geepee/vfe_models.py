"""Summary

"""
import numpy as np
from scipy.optimize import minimize
import pdb
from scipy.cluster.vq import kmeans2

from utils import *
from kernels import *
from config import *


class VI_Model(object):
    """Summary

    Attributes:
        fixed_params (list): Description
        N (TYPE): Description
        updated (bool): Description
        y_train (TYPE): Description
    """

    def __init__(self, y_train):
        """Summary

        Args:
            y_train (TYPE): Description
        """
        self.y_train = y_train
        self.N = y_train.shape[0]
        self.fixed_params = []
        self.updated = False

    def init_hypers(self, x_train=None):
        """Summary

        Args:
            x_train (None, optional): Description

        Returns:
            TYPE: Description
        """
        pass

    def get_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        pass

    def update_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        pass

    def optimise(
            self, method='L-BFGS-B', tol=None, reinit_hypers=True,
            callback=None, maxfun=100000, maxiter=1000, alpha=0.5,
            mb_size=None, adam_lr=0.001, **kargs):
        """Summary

        Args:
            method (str, optional): Description
            tol (None, optional): Description
            reinit_hypers (bool, optional): Description
            callback (None, optional): Description
            maxfun (int, optional): Description
            maxiter (int, optional): Description
            alpha (float, optional): Description
            mb_size (None, optional): Description
            adam_lr (float, optional): Description
            **kargs: Description

        Returns:
            TYPE: Description
        """
        self.updated = False

        if reinit_hypers:
            init_params_dict = self.init_hypers()
        else:
            init_params_dict = self.get_hypers()

        init_params_vec, params_args = flatten_dict(init_params_dict)
        objective_wrapper = ObjectiveWrapper()

        if mb_size is None:
            mb_size = self.N
        try:
            if method.lower() == 'adam':
                results = adam(objective_wrapper, init_params_vec,
                               step_size=adam_lr,
                               maxiter=maxiter,
                               args=(params_args, self, mb_size, alpha, None))
                final_params = results
            else:
                options = {'maxfun': maxfun, 'maxiter': maxiter,
                           'disp': True, 'gtol': 1e-8}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, self.N, alpha, None),
                    method=method,
                    jac=True,
                    tol=tol,
                    callback=callback,
                    options=options)
                final_params = results.x

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'
            final_params = objective_wrapper.previous_x
            # todo: deal with rresults here

        # results = self.get_hypers()
        final_params = unflatten_dict(final_params, params_args)
        self.update_hypers(final_params)
        return final_params

    def set_fixed_params(self, params):
        """Summary

        Args:
            params (TYPE): Description

        Returns:
            TYPE: Description
        """
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)


class SGPR(VI_Model):
    """Summary

    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        ls (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        sf (int): Description
        sn (int): Description
        updated (bool): Description
        x_train (TYPE): Description
        zu (TYPE): Description
    """

    def __init__(self, x_train, y_train, no_pseudo):
        '''
        This only works with real-valued (Gaussian) regression

        Args:
            x_train (TYPE): Description
            y_train (TYPE): Description
            no_pseudo (TYPE): Description

        '''

        super(SGPR, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = x_train.shape[1]
        self.M = M = no_pseudo
        self.x_train = x_train

        # variables for the hyperparameters
        self.zu = np.zeros([M, Din])
        self.ls = np.zeros([Din, ])
        self.sf = 0
        self.sn = 0

    def objective_function(self, params, idxs=None, alpha=1.0, prop_mode=None):
        """Summary

        Args:
            params (TYPE): Description
            idxs (None, optional): Description
            alpha (float, optional): Description
            prop_mode (None, optional): Description

        Returns:
            TYPE: Description
        """
        x = self.x_train
        y = self.y_train
        N = self.N
        Dout = self.Dout
        M = self.M
        # update model with new hypers
        self.update_hypers(params)
        sf2 = np.exp(2 * self.sf)
        sn2 = np.exp(2 * self.sn)

        # compute the approximate log marginal likelihood
        Kuu_noiseless = compute_kernel(
            2 * self.ls, 2 * self.sf, self.zu, self.zu)
        Kuu = Kuu_noiseless + np.diag(JITTER * np.ones((M, )))
        Lu = np.linalg.cholesky(Kuu)
        Kuf = compute_kernel(2 * self.ls, 2 * self.sf, self.zu, x)
        V = np.linalg.solve(Lu, Kuf)
        r = sf2 - np.sum(V * V, axis=0)
        G = alpha * r + sn2
        iG = 1.0 / G
        A = np.eye(M) + np.dot(V * iG, V.T)
        La = np.linalg.cholesky(A)
        B = np.linalg.solve(La, V)
        iGy = iG[:, None] * y
        z_tmp = np.dot(B.T, np.dot(B, iGy)) * iG[:, None]
        z = iGy - z_tmp
        term1 = 0.5 * np.sum(z * y)
        term2 = Dout * np.sum(np.log(np.diag(La)))
        term3 = 0.5 * Dout * np.sum(np.log(G))
        term4 = 0.5 * N * Dout * np.log(2 * np.pi)
        c4 = 0.5 * Dout * (1 - alpha) / alpha
        term5 = c4 * np.sum(np.log(1 + alpha * r / sn2))
        energy = term1 + term2 + term3 + term4 + term5
        # print term1, term2 + term3, term4, term5

        zus = self.zu / np.exp(self.ls)
        xs = x / np.exp(self.ls)
        R = np.linalg.solve(Lu.T, V)
        RiG = R * iG
        RdQ = -np.dot(np.dot(R, z), z.T) + RiG - \
            np.dot(np.dot(RiG, B.T), B) * iG
        dG = z**2 + (-iG + iG**2 * np.sum(B * B, axis=0))[:, None]
        tmp = alpha * dG - (1 - alpha) / (1 + alpha * r[:, None] / sn2) / sn2
        RdQ2 = RdQ + R * tmp.T
        KW = Kuf * RdQ2
        KWR = Kuu_noiseless * (np.dot(RdQ2, R.T))
        P = (np.dot(KW, xs) - np.dot(KWR, zus)
             + (np.sum(KWR, axis=1) - np.sum(KW, axis=1))[:, None] * zus)
        dzu = P / np.exp(self.ls)
        dls = (-np.sum(P * zus, axis=0)
               - np.sum((np.dot(KW.T, zus) - np.sum(KW, axis=0)[:, None] * xs) * xs, axis=0))
        dsn = -np.sum(dG) * sn2 - (1 - alpha) * \
            np.sum(r / (1 + alpha * r / sn2)) / sn2
        dsf = (np.sum(Kuf * RdQ) - alpha * np.sum(r[:, None] * dG)
               + (1 - alpha) * np.sum(r / (1 + alpha * r / sn2)) / sn2)

        grad_all = {'zu': dzu, 'ls': dls, 'sn': dsn, 'sf': dsf}
        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def predict_f(self, inputs, alpha=1.0, marginal=True):
        """Summary

        Args:
            inputs (TYPE): Description
            alpha (float, optional): Description
            marginal (bool, optional): Description

        Returns:
            TYPE: Description
        """
        x = self.x_train
        y = self.y_train
        N = self.N
        Dout = self.Dout
        M = self.M
        sf2 = np.exp(2 * self.sf)
        sn2 = np.exp(2 * self.sn)

        # compute the approximate log marginal likelihood
        Kuu_noiseless = compute_kernel(
            2 * self.ls, 2 * self.sf, self.zu, self.zu)
        Kuu = Kuu_noiseless + np.diag(JITTER * np.ones((M, )))
        Lu = np.linalg.cholesky(Kuu)
        Kuf = compute_kernel(2 * self.ls, 2 * self.sf, self.zu, x)
        Kut = compute_kernel(2 * self.ls, 2 * self.sf, self.zu, inputs)
        V = np.linalg.solve(Lu, Kuf)
        r = sf2 - np.sum(V * V, axis=0)
        G = alpha * r + sn2
        iG = 1.0 / G
        A = np.eye(M) + np.dot(V * iG, V.T)
        La = np.linalg.cholesky(A)
        B = np.linalg.solve(La, V)
        yhat = y * iG[:, None]
        yhatB = np.dot(yhat.T, B.T)
        beta = np.linalg.solve(Lu.T, np.linalg.solve(La.T, yhatB.T))
        W1 = np.eye(M) - np.linalg.solve(La.T,
                                         np.linalg.solve(La, np.eye(M))).T
        W = np.linalg.solve(Lu.T, np.linalg.solve(Lu.T, W1).T)
        KtuW = np.dot(Kut.T, W)
        mf = np.dot(Kut.T, beta)
        if marginal:
            vf = np.exp(2 * self.sf) - np.sum(KtuW * Kut.T, axis=1)
        else:
            Ktt = compute_kernel(2 * self.ls, 2 * self.sf, inputs, inputs)
            Ktt += np.diag(JITTER * np.ones((inputs.shape[0], )))
            vf = Ktt - np.dot(KtuW, Kut)
        return mf, vf

    def sample_f(self, inputs, no_samples=1):
        """Summary

        Args:
            inputs (TYPE): Description
            no_samples (int, optional): Description

        Returns:
            TYPE: Description
        """
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        K = no_samples
        fs = np.zeros((inputs.shape[0], self.Dout, K))
        # TODO: remove for loop here
        for k in range(K):
            fs[:, :, k] = self.sgp_layer.sample(inputs)
        return fs

    def predict_y(self, inputs):
        """Summary

        Args:
            inputs (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.output_probabilistic(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def init_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        N = self.N
        M = self.M
        Din = self.Din
        Dout = self.Dout
        x_train = self.x_train
        if N < 10000:
            centroids, label = kmeans2(x_train, M, minit='points')
        else:
            randind = np.random.permutation(N)
            centroids = x_train[randind[0:M], :]
        zu = centroids
        if N < 10000:
            X1 = np.copy(x_train)
        else:
            randind = np.random.permutation(N)
            X1 = X[randind[:5000], :]
        x_dist = cdist(X1, X1, 'euclidean')
        triu_ind = np.triu_indices(N)
        ls = np.zeros((Din, ))
        d2imed = np.median(x_dist[triu_ind])
        for i in range(Din):
            ls[i] = 2 * np.log(d2imed + 1e-16)
        sf = np.log(np.array([0.5]))

        params = dict()
        params['sf'] = sf
        params['ls'] = ls
        params['zu'] = zu
        params['sn'] = np.log(0.01)
        return params

    def get_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        params = {}
        M = self.M
        Din = self.Din
        Dout = self.Dout
        params['ls'] = self.ls
        params['sf'] = self.sf
        params['zu'] = self.zu
        params['sn'] = self.sn
        return params

    def update_hypers(self, params):
        """Summary

        Args:
            params (TYPE): Description

        Returns:
            TYPE: Description
        """
        self.ls = params['ls']
        self.sf = params['sf']
        self.zu = params['zu']
        self.sn = params['sn']
