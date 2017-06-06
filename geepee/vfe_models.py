"""Summary

"""
import numpy as np
from scipy.optimize import minimize
import pdb
from scipy.cluster.vq import kmeans2

from utils import *
from kernels import *
from config import *
from aep_models import Gauss_Layer, Probit_Layer


class VFE_Model(object):
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

    def init_hypers(self, y_train, x_train=None):
        """Summary

        Args:
            y_train (TYPE): Description
            x_train (None, optional): Description

        """
        pass

    def get_hypers(self):
        """Summary
        """
        pass

    def update_hypers(self, params):
        """Summary

        Args:
            params (TYPE): Description
        """
        pass

    def optimise(
            self, method='L-BFGS-B', tol=None, reinit_hypers=True,
            callback=None, maxfun=100000, maxiter=1000, alpha=0.5,
            mb_size=None, adam_lr=0.001, prop_mode=PROP_MM, disp=True, **kargs):
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
            prop_mode (TYPE, optional): Description
            disp (bool, optional): Description
            **kargs: Description

        Returns:
            TYPE: Description
        """
        self.updated = False

        if reinit_hypers:
            init_params_dict = self.init_hypers(self.y_train)
        else:
            init_params_dict = self.get_hypers()
        init_params_vec, params_args = flatten_dict(init_params_dict)
        objective_wrapper = ObjectiveWrapper()

        if mb_size is None:
            mb_size = self.N
        try:
            if method.lower() == 'adam':
                results = adam(
                    objective_wrapper, init_params_vec,
                    step_size=adam_lr,
                    maxiter=maxiter,
                    args=(params_args, self, mb_size, alpha, prop_mode),
                    disp=disp,
                    callback=callback)
                final_params = results
            else:
                options = {'maxfun': maxfun, 'maxiter': maxiter,
                           'disp': disp, 'gtol': 1e-8}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, self.N, alpha, prop_mode),
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
        """
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)

    def save_model(self, fname='/tmp/model.pickle'):
        """Summary

        Args:
            fname (str, optional): Description
        """
        params = self.get_hypers()
        pickle.dump(params, open(fname, "wb"))

    def load_model(self, fname='/tmp/model.pickle'):
        """Summary

        Args:
            fname (str, optional): Description
        """
        params = pickle.load(open(fname, "rb"))
        self.update_hypers(params)



# class VFE_Model(object):
#     """Summary

#     Attributes:
#         fixed_params (list): Description
#         N (TYPE): Description
#         updated (bool): Description
#         y_train (TYPE): Description
#     """

#     def __init__(self, y_train):
#         """Summary

#         Args:
#             y_train (TYPE): Description
#         """
#         self.y_train = y_train
#         self.N = y_train.shape[0]
#         self.fixed_params = []
#         self.updated = False

#     def init_hypers(self, y_train, x_train=None):
#         """Summary

#         Args:
#             y_train (TYPE): Description
#             x_train (None, optional): Description

#         """
#         pass

#     def get_hypers(self):
#         """Summary

#         Returns:
#             TYPE: Description
#         """
#         pass

#     def update_hypers(self):
#         """Summary

#         Returns:
#             TYPE: Description
#         """
#         pass

#     def optimise(
#             self, method='L-BFGS-B', alpha=0.01, tol=None, reinit_hypers=True,
#             callback=None, maxfun=100000, maxiter=1000,
#             mb_size=None, adam_lr=0.001, **kargs):
#         """Summary

#         Args:
#             method (str, optional): Description
#             tol (None, optional): Description
#             reinit_hypers (bool, optional): Description
#             callback (None, optional): Description
#             maxfun (int, optional): Description
#             maxiter (int, optional): Description
#             alpha (float, optional): Description
#             mb_size (None, optional): Description
#             adam_lr (float, optional): Description
#             **kargs: Description

#         Returns:
#             TYPE: Description
#         """
#         self.updated = False

#         if reinit_hypers:
#             init_params_dict = self.init_hypers(self.y_train)
#         else:
#             init_params_dict = self.get_hypers()

#         init_params_vec, params_args = flatten_dict(init_params_dict)
#         objective_wrapper = ObjectiveWrapper()

#         if mb_size is None:
#             mb_size = self.N
#         try:
#             if method.lower() == 'adam':
#                 results = adam(objective_wrapper, init_params_vec,
#                                step_size=adam_lr,
#                                maxiter=maxiter,
#                                args=(params_args, self, mb_size, alpha, None))
#                 final_params = results
#             else:
#                 options = {'maxfun': maxfun, 'maxiter': maxiter,
#                            'disp': True, 'gtol': 1e-8}
#                 results = minimize(
#                     fun=objective_wrapper,
#                     x0=init_params_vec,
#                     args=(params_args, self, self.N, alpha, None),
#                     method=method,
#                     jac=True,
#                     tol=tol,
#                     callback=callback,
#                     options=options)
#                 final_params = results.x

#         except KeyboardInterrupt:
#             print 'Caught KeyboardInterrupt ...'
#             final_params = objective_wrapper.previous_x
#             # todo: deal with rresults here

#         # results = self.get_hypers()
#         final_params = unflatten_dict(final_params, params_args)
#         self.update_hypers(final_params)
#         return final_params

#     def set_fixed_params(self, params):
#         """Summary

#         Args:
#             params (TYPE): Description

#         Returns:
#             TYPE: Description
#         """
#         if isinstance(params, (list)):
#             for p in params:
#                 if p not in self.fixed_params:
#                     self.fixed_params.append(p)
#         else:
#             self.fixed_params.append(params)

#     def save_model(self, fname='/tmp/model.pickle'):
#         """Summary

#         Args:
#             fname (str, optional): Description

#         Returns:
#             TYPE: Description
#         """
#         params = self.get_hypers()
#         pickle.dump(params, open(fname, "wb"))

#     def load_model(self, fname='/tmp/model.pickle'):
#         """Summary

#         Args:
#             fname (str, optional): Description

#         Returns:
#             TYPE: Description
#         """
#         params = pickle.load(open(fname, "rb"))
#         self.update_hypers(params)


class SGPR_collapsed(VFE_Model):
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

        super(SGPR_collapsed, self).__init__(y_train)
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

    def predict_f(self, inputs, alpha=0.01, marginal=True):
        """Summary

        Args:
            inputs (TYPE): Description
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
        # TODO
        """Summary

        Args:
            inputs (TYPE): Description
            no_samples (int, optional): Description

        Returns:
            TYPE: Description
        """
        K = no_samples
        fs = np.zeros((inputs.shape[0], self.Dout, K))
        # TODO: remove for loop here
        for k in range(K):
            fs[:, :, k] = self.sgp_layer.sample(inputs)
        return fs

    def predict_y(self, inputs, alpha=0.01, marginal=True):
        """Summary

        Args:
            inputs (TYPE): Description

        Returns:
            TYPE: Description
        """
        mf, vf = self.predict_f(inputs, alpha, marginal)
        if marginal:
            my, vy = mf, vf + np.exp(2 * self.sn)
        else:
            my, vy = mf, vf + np.exp(2 * self.sn) * np.eye(my.shape[0])
        return my, vy

    def init_hypers(self, y_train):
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


class SGP_Layer(object):
    """Sparse Gaussian process layer
    """

    def __init__(self, no_train, input_size, output_size, no_pseudo):
        """Initialisation

        Args:
            no_train (int): Number of training points
            input_size (int): Number of input dimensions
            output_size (int): Number of output dimensions
            no_pseudo (int): Number of pseudo-points
        """
        self.Din = Din = input_size
        self.Dout = Dout = output_size
        self.M = M = no_pseudo
        self.N = N = no_train

        self.ones_M = np.ones(M)
        self.ones_Din = np.ones(Din)

        # variables for the mean and covariance of q(u)
        self.mu = np.zeros([Dout, M, ])
        self.Su = np.zeros([Dout, M, M])
        self.Splusmm = np.zeros([Dout, M, M])

        # numpy variable for inducing points, Kuuinv, Kuu and its gradients
        self.zu = np.zeros([M, Din])
        self.Kuu = np.zeros([M, M])
        self.Kuuinv = np.zeros([M, M])

        # variables for the hyperparameters
        self.ls = np.zeros([Din, ])
        self.sf = 0

        # and natural parameters
        self.theta_1_R = np.zeros([Dout, M, M])
        self.theta_2 = np.zeros([Dout, M, ])
        self.theta_1 = np.zeros([Dout, M, M])

        # terms that are common to all datapoints in each minibatch
        self.A = np.zeros([Dout, M, ])
        self.B_det = np.zeros([Dout, M, M])
        self.B_sto = np.zeros([Dout, M, M])

    def compute_KL(self):
        # log det prior
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        log_det_prior = self.Dout * logdet
        # log det posterior
        (sign, logdet) = np.linalg.slogdet(self.Su)
        log_det_post = np.sum(logdet)
        # trace term
        trace_term = np.sum(self.Kuuinv * self.Splusmm)
        KL = 0.5 * (log_det_prior - log_det_post - self.Dout * self.M + trace_term)
        return KL

    def forward_prop_thru_post(self, mx, vx=None, mode=PROP_MM):
        """Propagate input distributions through the  non-linearity

        Args:
            mx (float): means of the input distributions, size K x Din
            vx (float, optional): variances (if uncertain inputs), size K x Din
            mode (config param, optional): propagation mode (see config)

        Returns:
            specific results depend on the propagation mode provided

        Raises:
            NotImplementedError: Unknown propagation mode
        """
        if vx is None:
            return self._forward_prop_deterministic_thru_post(mx)
        else:
            if mode == PROP_MM:
                return self._forward_prop_random_thru_post_mm(mx, vx)
            elif mode == PROP_LIN:
                return self._forward_prop_random_thru_post_lin(mx, vx)
            elif mode == PROP_MC:
                return self._forward_prop_random_thru_post_mc(mx, vx)
            else:
                raise NotImplementedError('unknown propagation mode')

    def _forward_prop_deterministic_thru_post(self, x):
        """Propagate deterministic inputs thru cavity

        Args:
            x (float): input values, size K x Din

        Returns:
            float, size K x Dout: output means
            float, size K x Dout: output variances
            float, size K x M: cross covariance matrix
        """
        kff = np.exp(2 * self.sf)
        kfu = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', kfu, self.A)
        Bkfukuf = np.einsum('dab,na,nb->nd', self.B_det, kfu, kfu)
        vout = kff + Bkfukuf
        return mout, vout, kfu

    def _forward_prop_random_thru_post_mc(self, mx, vx):
        """Propagate uncertain inputs thru cavity, using simple Monte Carlo

        Args:
            mx (float): input means, size K x Din
            vx (TYPE): input variances, size K x Din

        Returns:
            output means and variances, and intermediate info for backprop
        """
        batch_size = mx.shape[0]
        eps = np.random.randn(MC_NO_SAMPLES, batch_size, self.Din)
        x = eps * np.sqrt(vx) + mx
        x_stk = np.reshape(x, [MC_NO_SAMPLES * batch_size, self.Din])
        e_stk = np.reshape(eps, [MC_NO_SAMPLES * batch_size, self.Din])
        m_stk, v_stk, kfu_stk = self._forward_prop_deterministic_thru_post(
            x_stk)
        mout = m_stk.reshape([MC_NO_SAMPLES, batch_size, self.Dout])
        vout = v_stk.reshape([MC_NO_SAMPLES, batch_size, self.Dout])
        kfu = kfu_stk.reshape([MC_NO_SAMPLES, batch_size, self.M])
        return (mout, vout, kfu, x, eps), (m_stk, v_stk, kfu_stk, x_stk, e_stk)

    @profile
    def _forward_prop_random_thru_post_mm(self, mx, vx):
        """Propagate uncertain inputs thru cavity, using simple Moment Matching

        Args:
            mx (float): input means, size K x Din
            vx (TYPE): input variances, size K x Din

        Returns:
            output means and variances, and intermediate info for backprop
        """
        psi0 = np.exp(2 * self.sf)
        psi1, psi2 = compute_psi_weave(
            2 * self.ls, 2 * self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.B_sto, psi2)
        vout = psi0 + Bhatpsi2 - mout**2
        return mout, vout, psi1, psi2

    def forward_prop_thru_cav(self, mx, vx=None, mode=PROP_MM):
        """Propagate input distributions through the cavity non-linearity

        Args:
            mx (float): means of the input distributions, size K x Din
            vx (float, optional): variances (if uncertain inputs), size K x Din
            mode (config param, optional): propagation mode (see config)

        Returns:
            specific results depend on the propagation mode provided

        Raises:
            NotImplementedError: Unknown propagation mode
        """
        if vx is None:
            return self._forward_prop_deterministic_thru_cav(mx)
        else:
            if mode == PROP_MM:
                return self._forward_prop_random_thru_cav_mm(mx, vx)
            elif mode == PROP_LIN:
                return self._forward_prop_random_thru_cav_lin(mx, vx)
            elif mode == PROP_MC:
                return self._forward_prop_random_thru_cav_mc(mx, vx)
            else:
                raise NotImplementedError('unknown propagation mode')

    def _forward_prop_deterministic_thru_cav(self, x):
        """Propagate deterministic inputs thru cavity

        Args:
            x (float): input values, size K x Din

        Returns:
            float, size K x Dout: output means
            float, size K x Dout: output variances
            float, size K x M: cross covariance matrix
        """
        kff = np.exp(2 * self.sf)
        kfu = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', kfu, self.Ahat)
        Bkfukuf = np.einsum('dab,na,nb->nd', self.Bhat_det, kfu, kfu)
        vout = kff + Bkfukuf
        return mout, vout, kfu

    def _forward_prop_random_thru_cav_mc(self, mx, vx):
        """Propagate uncertain inputs thru cavity, using simple Monte Carlo

        Args:
            mx (float): input means, size K x Din
            vx (TYPE): input variances, size K x Din

        Returns:
            output means and variances, and intermediate info for backprop
        """
        batch_size = mx.shape[0]
        eps = np.random.randn(MC_NO_SAMPLES, batch_size, self.Din)
        x = eps * np.sqrt(vx) + mx
        x_stk = np.reshape(x, [MC_NO_SAMPLES * batch_size, self.Din])
        e_stk = np.reshape(eps, [MC_NO_SAMPLES * batch_size, self.Din])
        m_stk, v_stk, kfu_stk = self._forward_prop_deterministic_thru_cav(
            x_stk)
        mout = m_stk.reshape([MC_NO_SAMPLES, batch_size, self.Dout])
        vout = v_stk.reshape([MC_NO_SAMPLES, batch_size, self.Dout])
        kfu = kfu_stk.reshape([MC_NO_SAMPLES, batch_size, self.M])
        return (mout, vout, kfu, x, eps), (m_stk, v_stk, kfu_stk, x_stk, e_stk)

    @profile
    def _forward_prop_random_thru_cav_mm(self, mx, vx):
        """Propagate uncertain inputs thru cavity, using simple Moment Matching

        Args:
            mx (float): input means, size K x Din
            vx (TYPE): input variances, size K x Din

        Returns:
            output means and variances, and intermediate info for backprop
        """
        psi0 = np.exp(2 * self.sf)
        psi1, psi2 = compute_psi_weave(
            2 * self.ls, 2 * self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.Ahat)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.Bhat_sto, psi2)
        vout = psi0 + Bhatpsi2 - mout**2
        return mout, vout, psi1, psi2


    #TODO
    @profile
    def backprop_grads_lvm_mm(self, m, v, dm, dv, psi1, psi2, mx, vx):
        """Summary

        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm (TYPE): Description
            dv (TYPE): Description
            psi1 (TYPE): Description
            psi2 (TYPE): Description
            mx (TYPE): Description
            vx (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2 * self.sf)
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        mu = self.mu
        Su = self.Su
        Spmm = self.Splusmm
        Kuuinv = self.Kuuinv
        Kuu = self.Kuu

        # compute grads wrt psi1 and psi2
        dm_all = dm - 2 * dv * m
        dpsi1 = np.einsum('nd,dm->nm', dm_all, self.A)
        dpsi2 = np.einsum('nd,dab->nab', dv, self.B_sto)
        dsf2, dls, dzu, dmx, dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, ls, sf2, mx, vx, self.zu)

        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        # compute grads wrt theta1 and theta2 via log lik exp term
        dA = np.einsum('nd,nm->dm', dm_all, psi1)
        dB = np.einsum('nd,nab->dab', dv, psi2)
        dSu_via_v = np.einsum('ab,dbc,ce->dae', Kuuinv, dB, Kuuinv)
        dmu = 2 * np.einsum('dab,db->da', dSu_via_v, mu) \
            + np.einsum('ab,db->da', Kuuinv, dA)

        dSu_via_m = np.einsum('da,db->dab', dmu, self.theta_2)
        dSu = dSu_via_m + dSu_via_v
        dSuinv = - np.einsum('dab,dbc,dce->dae', Su, dSu, Su)
        dtheta1 = dSuinv
        dtheta2 = np.einsum('dab,db->da', Su, dmu)
        # add contrib from the KL term
        dtheta2 += np.einsum('dab,bc,dc->da', Su, Kuuinv, mu)
        dtheta1_1 = 0.5 * Su
        SuKuuinv = np.einsum('dab,bc->dac', Su, Kuuinv)
        dtheta1_2 = - 0.5 * np.einsum('dab,dbc->dac', SuKuuinv, Su)
        mutheta2Su = np.einsum('da,db,dbc->dac', mu, self.theta_2, Su)
        dtheta1_3 = - np.einsum('dab,dbc->dac', SuKuuinv, mutheta2Su)
        dtheta1 += dtheta1_1 + dtheta1_2 + dtheta1_3
        dtheta1T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum(
            'dab,dbc->dac', self.theta_1_R, dtheta1 + dtheta1T)

        deta1_R = np.zeros([self.Dout, M * (M + 1) / 2])
        deta2 = dtheta2
        for d in range(self.Dout):
            dtheta1_R_d = dtheta1_R[d, :, :]
            theta1_R_d = self.theta_1_R[d, :, :]
            dtheta1_R_d[diag_ind] = dtheta1_R_d[
                diag_ind] * theta1_R_d[diag_ind]
            dtheta1_R_d = dtheta1_R_d[triu_ind]
            deta1_R[d, :] = dtheta1_R_d.reshape(
                (dtheta1_R_d.shape[0], ))

        # grads wrt Kuu
        dKuuinv_Su = np.sum(dtheta1, axis=0)
        dKuuinv_KL = - 0.5 * self.Dout * Kuu + 0.5 * np.sum(Spmm, axis=0)
        dKuuinv_A = np.einsum('da,db->ab', dA, mu)
        KuuinvSmmd = np.einsum('ab,dbc->dac', Kuuinv, Spmm)
        dKuuinv_B = 2 * np.einsum('dab,dac->bc', KuuinvSmmd, dB) \
            - np.sum(dB, axis=0)
        dKuuinv = dKuuinv_A + dKuuinv_B + dKuuinv_Su + dKuuinv_KL
        M_inner = - np.dot(Kuuinv, np.dot(dKuuinv, Kuuinv))
        dhyp = d_trace_MKzz_dhypers(
            2 * self.ls, 2 * self.sf, self.zu, M_inner,
            self.Kuu - np.diag(JITTER * np.ones(self.M)))

        dzu += dhyp[2]
        dls += 2 * dhyp[1]
        dsf += 2 * dhyp[0]

        grad_hyper = {
            'sf': dsf, 'ls': dls, 'zu': dzu,
            'eta1_R': deta1_R, 'eta2': deta2}
        grad_input = {'mx': dmx, 'vx': dvx}

        return grad_hyper, grad_input

    #TODO
    @profile
    def backprop_grads_lvm_mc(self, m, v, dm, dv, kfu, x, alpha=1.0):
        """Summary

        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm (TYPE): Description
            dv (TYPE): Description
            kfu (TYPE): Description
            x (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2 * self.sf)
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        mu = self.mu
        Su = self.Su
        Spmm = self.Splusmm
        muhat = self.muhat
        Suhat = self.Suhat
        Spmmhat = self.Splusmmhat
        Kuuinv = self.Kuuinv
        Kuu = self.Kuu
        kfuKuuinv = np.dot(kfu, Kuuinv)
        dm = dm.reshape(m.shape)
        dv = dv.reshape(v.shape)

        beta = (N - alpha) * 1.0 / N
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1

        # compute grads wrt kfu
        dkfu_m = np.einsum('nd,dm->nm', dm, self.Ahat)
        dkfu_v = 2 * np.einsum('nd,dab,na->nb', dv, self.Bhat_det, kfu)
        dkfu = dkfu_m + dkfu_v
        dsf2, dls, dzu, dx = compute_kfu_derivatives(
            dkfu, kfu, ls, sf2, x, self.zu, grad_x=True)
        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        # compute grads wrt theta1 and theta2
        SKuuinvKuf = np.einsum('dab,nb->nda', Suhat, kfuKuuinv)
        dSinv_via_v = - np.einsum('nda,nd,ndb->dab',
                                  SKuuinvKuf, dv, SKuuinvKuf)
        dSinv_via_m = - np.einsum('nda,nd,db->dab', SKuuinvKuf, dm, muhat)
        dSinv = dSinv_via_m + dSinv_via_v
        dSinvM = np.einsum('nda,nd->da', SKuuinvKuf, dm)
        dtheta1 = beta * dSinv
        dtheta2 = beta * dSinvM

        dtheta1 = -0.5 * scale_poste * Spmm - 0.5 * scale_cav * beta * Spmmhat + dtheta1
        dtheta2 = scale_poste * mu + scale_cav * beta * muhat + dtheta2
        dtheta1T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum(
            'dab,dbc->dac', self.theta_1_R, dtheta1 + dtheta1T)

        deta1_R = np.zeros([self.Dout, M * (M + 1) / 2])
        deta2 = dtheta2
        for d in range(self.Dout):
            dtheta1_R_d = dtheta1_R[d, :, :]
            theta1_R_d = self.theta_1_R[d, :, :]
            dtheta1_R_d[diag_ind] = dtheta1_R_d[
                diag_ind] * theta1_R_d[diag_ind]
            dtheta1_R_d = dtheta1_R_d[triu_ind]
            deta1_R[d, :] = dtheta1_R_d.reshape(
                (dtheta1_R_d.shape[0], ))

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dAhat = np.einsum('nd,nm->dm', dm, kfu)
        dKuuinv_m = np.einsum('da,db->ab', dAhat, muhat)
        KuuinvSmmd = np.einsum('ab,dbc->dac', Kuuinv, Suhat)
        dBhat = np.einsum('nd,na,nb->dab', dv, kfu, kfu)
        dKuuinv_v_1 = 2 * np.einsum('dab,dac->bc', KuuinvSmmd, dBhat) \
            - np.sum(dBhat, axis=0)
        dKuuinv_v_2 = np.sum(dSinv, axis=0)
        dKuuinv = dKuuinv_m + dKuuinv_v_1 + dKuuinv_v_2

        Minner = scale_poste * np.sum(Spmm, axis=0) + scale_cav * \
            np.sum(Spmmhat, axis=0) - 2.0 * dKuuinv
        M_all = 0.5 * (scale_prior * self.Dout * Kuuinv +
                       np.dot(Kuuinv, np.dot(Minner, Kuuinv)))
        dhyp = d_trace_MKzz_dhypers(
            2 * self.ls, 2 * self.sf, self.zu, M_all,
            self.Kuu - np.diag(JITTER * np.ones(self.M)))

        dzu += dhyp[2]
        dls += 2 * dhyp[1]
        dsf += 2 * dhyp[0]

        grad_hyper = {
            'sf': dsf, 'ls': dls, 'zu': dzu,
            'eta1_R': deta1_R, 'eta2': deta2
        }

        return grad_hyper, dx

    def backprop_grads_reparam(self, dx, m, v, eps):
        """Summary

        Args:
            dx (TYPE): Description
            m (TYPE): Description
            v (TYPE): Description
            eps (TYPE): Description

        Returns:
            TYPE: Description
        """
        dx = dx.reshape(eps.shape)
        dm = np.sum(dx, axis=0)
        dv = np.sum(dx * eps, axis=0) / (2 * np.sqrt(v))
        return {'mx': dm, 'vx': dv}

    @profile
    def backprop_grads_reg(self, m, v, dm, dv, kfu, x):
        """Summary

        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm (TYPE): Description
            dv (TYPE): Description
            kfu (TYPE): Description
            x (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2 * self.sf)
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        mu = self.mu
        Su = self.Su
        Spmm = self.Splusmm
        Kuuinv = self.Kuuinv
        Kuu = self.Kuu
        kfuKuuinv = np.dot(kfu, Kuuinv)

        # compute grads wrt kfu
        dkfu_m = np.einsum('nd,dm->nm', dm, self.A)
        dkfu_v = 2 * np.einsum('nd,dab,na->nb', dv, self.B_det, kfu)
        dkfu = dkfu_m + dkfu_v
        dsf2, dls, dzu = compute_kfu_derivatives(
            dkfu, kfu, ls, sf2, x, self.zu)
        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        # compute grads wrt theta1 and theta2 via log lik exp term
        SKuuinvKuf = np.einsum('dab,nb->nda', Su, kfuKuuinv)
        dSinv_via_v = - np.einsum('nda,nd,ndb->dab',
                                  SKuuinvKuf, dv, SKuuinvKuf)
        dSinv_via_m = - np.einsum('nda,nd,db->dab', SKuuinvKuf, dm, mu)
        dtheta1 = dSinv_via_m + dSinv_via_v
        dtheta2 = np.einsum('nda,nd->da', SKuuinvKuf, dm)
        # add in contrib from the KL term
        dtheta2 += np.einsum('dab,bc,dc->da', Su, Kuuinv, mu)
        dtheta1_1 = 0.5 * Su
        SuKuuinv = np.einsum('dab,bc->dac', Su, Kuuinv)
        dtheta1_2 = - 0.5 * np.einsum('dab,dbc->dac', SuKuuinv, Su)
        mutheta2Su = np.einsum('da,db,dbc->dac', mu, self.theta_2, Su)
        dtheta1_3 = - np.einsum('dab,dbc->dac', SuKuuinv, mutheta2Su)
        dtheta1 += dtheta1_1 + dtheta1_2 + dtheta1_3
        dtheta1T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum(
            'dab,dbc->dac', self.theta_1_R, dtheta1 + dtheta1T)

        deta1_R = np.zeros([self.Dout, M * (M + 1) / 2])
        deta2 = dtheta2
        for d in range(self.Dout):
            dtheta1_R_d = dtheta1_R[d, :, :]
            theta1_R_d = self.theta_1_R[d, :, :]
            dtheta1_R_d[diag_ind] = dtheta1_R_d[
                diag_ind] * theta1_R_d[diag_ind]
            dtheta1_R_d = dtheta1_R_d[triu_ind]
            deta1_R[d, :] = dtheta1_R_d.reshape(
                (dtheta1_R_d.shape[0], ))

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dA = np.einsum('nd,nm->dm', dm, kfu)
        dKuuinv_m = np.einsum('da,db->ab', dA, mu)
        KuuinvS = np.einsum('ab,dbc->dac', Kuuinv, Su)
        dB = np.einsum('nd,na,nb->dab', dv, kfu, kfu)
        dKuuinv_v = 2 * np.einsum('dab,dac->bc', KuuinvS, dB) \
            - np.sum(dB, axis=0)
        dKuuinv_S = np.sum(dtheta1, axis=0)
        dKuuinv_KL = - 0.5 * self.Dout * Kuu + 0.5 * np.sum(Spmm, axis=0)
        dKuuinv = dKuuinv_m + dKuuinv_v + dKuuinv_S + dKuuinv_KL
        M_inner = - np.dot(Kuuinv, np.dot(dKuuinv, Kuuinv))
        dhyp = d_trace_MKzz_dhypers(
            2 * self.ls, 2 * self.sf, self.zu, M_inner,
            self.Kuu - np.diag(JITTER * np.ones(self.M)))

        dzu += dhyp[2]
        dls += 2 * dhyp[1]
        dsf += 2 * dhyp[0]

        grad_hyper = {
            'sf': dsf, 'ls': dls, 'zu': dzu,
            'eta1_R': deta1_R, 'eta2': deta2
        }

        return grad_hyper

    def sample(self, x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        Su = self.Su
        mu = self.mu
        Lu = np.linalg.cholesky(Su)
        epsilon = np.random.randn(self.Dout, self.M)
        u_sample = mu + np.einsum('dab,db->da', Lu, epsilon)

        kff = compute_kernel(2 * self.ls, 2 * self.sf, x, x)
        kff += np.diag(JITTER * np.ones(x.shape[0]))
        kfu = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        qfu = np.dot(kfu, self.Kuuinv)
        mf = np.einsum('nm,dm->nd', qfu, u_sample)
        vf = kff - np.dot(qfu, kfu.T)
        Lf = np.linalg.cholesky(vf)
        epsilon = np.random.randn(x.shape[0], self.Dout)
        f_sample = mf + np.einsum('ab,bd->ad', Lf, epsilon)
        return f_sample

    def compute_kuu(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # update kuu and kuuinv
        ls = self.ls
        sf = self.sf
        Dout = self.Dout
        M = self.M
        zu = self.zu
        self.Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        self.Kuu += np.diag(JITTER * np.ones((M, )))
        # self.Kuuinv = matrixInverse(self.Kuu)
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def update_posterior(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # compute the posterior approximation
        Sinv = self.Kuuinv + self.theta_1
        self.Su = np.linalg.inv(Sinv)
        self.mu = np.einsum('dab,db->da', self.Su, self.theta_2)
        self.Splusmm = self.Su + np.einsum('da,db->dab', self.mu, self.mu)
        self.A = np.einsum('ab,db->da', self.Kuuinv, self.mu)
        self.B_sto = - self.Kuuinv + np.einsum(
            'ab,dbc->dac',
            self.Kuuinv,
            np.einsum('dab,bc->dac', self.Splusmm, self.Kuuinv))
        self.B_det = - self.Kuuinv + np.einsum(
            'ab,dbc->dac',
            self.Kuuinv,
            np.einsum('dab,bc->dac', self.Su, self.Kuuinv))


    def init_hypers(self, x_train=None, key_suffix=''):
        """Summary

        Args:
            x_train (None, optional): Description
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
        N = self.N
        M = self.M
        Din = self.Din
        Dout = self.Dout

        if x_train is None:
            ls = np.log(np.ones((Din, )) + 0.1 * np.random.rand(Din, ))
            sf = np.log(np.array([1]))
            zu = np.tile(np.linspace(-1, 1, M).reshape((M, 1)), (1, Din))
            # zu += 0.01 * np.random.randn(zu.shape[0], zu.shape[1])
        else:
            if N < 10000:
                centroids, label = kmeans2(x_train, M, minit='points')
            else:
                randind = np.random.permutation(N)
                centroids = x_train[randind[0:M], :]
            zu = centroids

            if N < 1000:
                X1 = np.copy(x_train)
            else:
                randind = np.random.permutation(N)
                X1 = x_train[randind[:1000], :]

            x_dist = cdist(X1, X1, 'euclidean')
            triu_ind = np.triu_indices(X1.shape[0])
            ls = np.zeros((Din, ))
            d2imed = np.median(x_dist[triu_ind])
            for i in range(Din):
                ls[i] = np.log(d2imed / 2 + 1e-16)
            sf = np.log(np.array([0.5]))

        Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        Kuu += np.diag(JITTER * np.ones((M, )))
        Kuuinv = matrixInverse(Kuu)

        eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        eta2 = np.zeros((Dout, M))
        for d in range(Dout):
            mu = np.linspace(-1, 1, M).reshape((M, 1))
            # mu += 0.01 * np.random.randn(M, 1)
            alpha = 0.5 * np.random.rand(M)
            # alpha = 0.01 * np.ones(M)
            Su = np.diag(alpha)
            Suinv = np.diag(1 / alpha)

            theta2 = np.dot(Suinv, mu)
            theta1 = Suinv
            R = np.linalg.cholesky(theta1).T

            triu_ind = np.triu_indices(M)
            diag_ind = np.diag_indices(M)
            R[diag_ind] = np.log(R[diag_ind])
            np.log(R[diag_ind])
            eta1_d = R[triu_ind].reshape((M * (M + 1) / 2,))
            eta2_d = theta2.reshape((M,))
            eta1_R[d, :] = eta1_d
            eta2[d, :] = eta2_d

        params = dict()
        params['sf' + key_suffix] = sf
        params['ls' + key_suffix] = ls
        params['zu' + key_suffix] = zu
        params['eta1_R' + key_suffix] = eta1_R
        params['eta2' + key_suffix] = eta2

        return params

    def get_hypers(self, key_suffix=''):
        """Summary

        Args:
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
        params = {}
        M = self.M
        Din = self.Din
        Dout = self.Dout
        params['ls' + key_suffix] = self.ls
        params['sf' + key_suffix] = self.sf
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        params_eta2 = self.theta_2
        params_eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        params_zu_i = self.zu

        for d in range(Dout):
            Rd = self.theta_1_R[d, :, :]
            Rd[diag_ind] = np.log(Rd[diag_ind])
            params_eta1_R[d, :] = Rd[triu_ind]

        params['zu' + key_suffix] = self.zu
        params['eta1_R' + key_suffix] = params_eta1_R
        params['eta2' + key_suffix] = params_eta2
        return params

    def update_hypers(self, params, key_suffix=''):
        """Summary

        Args:
            params (TYPE): Description
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
        M = self.M
        self.ls = params['ls' + key_suffix]
        self.sf = params['sf' + key_suffix]
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        zu = params['zu' + key_suffix]
        self.zu = zu

        for d in range(self.Dout):
            theta_m_d = params['eta2' + key_suffix][d, :]
            theta_R_d = params['eta1_R' + key_suffix][d, :]
            R = np.zeros((M, M))
            R[triu_ind] = theta_R_d.reshape(theta_R_d.shape[0], )
            R[diag_ind] = np.exp(R[diag_ind])
            self.theta_1_R[d, :, :] = R
            self.theta_1[d, :, :] = np.dot(R.T, R)
            self.theta_2[d, :] = theta_m_d

        # update Kuu given new hypers
        self.compute_kuu()
        # compute mu and Su for each layer
        self.update_posterior()


# TODO probit
class SGPR(VFE_Model):
    """Uncollapsed sparse Gaussian process approximations
    """

    def __init__(self, x_train, y_train, no_pseudo, lik='Gaussian'):
        """Summary

        Args:
            x_train (TYPE): Description
            y_train (TYPE): Description
            no_pseudo (TYPE): Description
            lik (str, optional): Description

        Raises:
            NotImplementedError: Description
        """
        super(SGPR, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = x_train.shape[1]
        self.M = M = no_pseudo
        self.x_train = x_train

        self.sgp_layer = SGP_Layer(N, Din, Dout, M)
        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

    @profile
    def objective_function(self, params, mb_size, alpha='not_used', prop_mode='not_used'):
        """Summary

        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            prop_mode (TYPE, optional): Description

        Returns:
            TYPE: Description
        """
        N = self.N
        if mb_size >= N:
            xb = self.x_train
            yb = self.y_train
        else:
            idxs = np.random.choice(N, mb_size, replace=False)
            xb = self.x_train[idxs, :]
            yb = self.y_train[idxs, :]
        batch_size = yb.shape[0]
        scale_loglik = - N * 1.0 / batch_size
        # scale_loglik = 0

        # update model with new hypers
        self.update_hypers(params)

        # propagate x cavity forward
        mout, vout, kfu = self.sgp_layer.forward_prop_thru_post(xb)
        # compute logZ and gradients
        logZ, dm, dv = self.lik_layer.compute_log_lik_exp(mout, vout, yb)
        logZ_scale = scale_loglik * logZ
        dm_scale = scale_loglik * dm
        dv_scale = scale_loglik * dv
        sgp_grad_hyper = self.sgp_layer.backprop_grads_reg(
            mout, vout, dm_scale, dv_scale, kfu, xb)
        lik_grad_hyper = self.lik_layer.backprop_grads_log_lik_exp(
            mout, vout, dm, dv, yb, scale_loglik)

        grad_all = {}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute objective
        sgp_KL_term = self.sgp_layer.compute_KL()
        energy = logZ_scale + sgp_KL_term
        # energy = logZ_scale
        # energy = sgp_KL_term

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def predict_f(self, inputs):
        """Summary

        Args:
            inputs (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not self.updated:
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf, _ = self.sgp_layer.forward_prop_thru_post(inputs)
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
            self.sgp_layer.update_posterior()
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
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf, _ = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def init_hypers(self, y_train):
        """Summary

        Args:
            y_train (TYPE): Description

        Returns:
            TYPE: Description
        """
        sgp_params = self.sgp_layer.init_hypers(self.x_train)
        lik_params = self.lik_layer.init_hypers()
        init_params = dict(sgp_params)
        init_params.update(lik_params)
        return init_params

    def get_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        sgp_params = self.sgp_layer.get_hypers()
        lik_params = self.lik_layer.get_hypers()
        params = dict(sgp_params)
        params.update(lik_params)
        return params

    def update_hypers(self, params):
        """Summary

        Args:
            params (TYPE): Description

        Returns:
            TYPE: Description
        """
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)


class SGPLVM(VFE_Model):
    """Summary

    """

    def __init__(self, y_train, hidden_size, no_pseudo,
                 lik='Gaussian', prior_mean=0, prior_var=1):
        """Summary

        Args:
            y_train (TYPE): Description
            hidden_size (TYPE): Description
            no_pseudo (TYPE): Description
            lik (str, optional): Description
            prior_mean (int, optional): Description
            prior_var (int, optional): Description

        Raises:
            NotImplementedError: Description
        """
        super(SGPLVM, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = hidden_size
        self.M = M = no_pseudo

        self.sgp_layer = SGP_Layer(N, Din, Dout, M)
        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

        # natural params for latent variables
        self.factor_x1 = np.zeros((N, Din))
        self.factor_x2 = np.zeros((N, Din))

        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.prior_x1 = prior_mean / prior_var
        self.prior_x2 = 1.0 / prior_var

        self.x_post_1 = np.zeros((N, Din))
        self.x_post_2 = np.zeros((N, Din))

    @profile
    def objective_function(self, params, mb_size, alpha='not_used', prop_mode=PROP_MM):
        """Summary

        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            alpha (float, optional): Description
            prop_mode (TYPE, optional): Description

        Returns:
            TYPE: Description

        Raises:
            NotImplementedError: Description
        """
        N = self.N
        sgp_layer = self.sgp_layer
        lik_layer = self.lik_layer
        if mb_size == N:
            idxs = np.arange(N)
            yb = self.y_train
        else:
            idxs = np.random.choice(N, mb_size, replace=False)
            yb = self.y_train[idxs, :]
        batch_size = yb.shape[0]
        scale_log_lik = - N * 1.0 / batch_size
        # scale_log_lik = 0

        # update model with new hypers
        self.update_hypers(params)

        # compute cavity
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]
        post_t1 = t01 + t11
        post_t2 = t02 + t12
        vx = 1.0 / post_t2
        mx = post_t1 / post_t2
        if prop_mode == PROP_MM:
            # propagate x cavity forward
            mout, vout, psi1, psi2 = sgp_layer.forward_prop_thru_post(mx, vx)
            # compute logZ and gradients
            logZ, dm, dv = self.lik_layer.compute_log_lik_exp(mout, vout, yb)    
            logZ_scale = scale_log_lik * logZ
            dm_scale = scale_log_lik * dm
            dv_scale = scale_log_lik * dv
            sgp_grad_hyper, sgp_grad_input = sgp_layer.backprop_grads_lvm_mm(
                mout, vout, dm_scale, dv_scale, psi1, psi2, mx, vx)
            lik_grad_hyper = self.lik_layer.backprop_grads_log_lik_exp(
                mout, vout, dm, dv, yb, scale_log_lik)
        elif prop_mode == PROP_MC:
            # TODO
            # propagate x cavity forward
            res, res_s = sgp_layer.forward_prop_thru_cav(mx, vx, PROP_MC)
            m, v, kfu, x, eps = res[0], res[1], res[2], res[3], res[4]
            m_s, v_s, kfu_s, x_s, eps_s = (
                res_s[0], res_s[1], res_s[2], res_s[3], res_s[4])
            # compute logZ and gradients
            logZ, dm, dv = lik_layer.compute_log_Z(m, v, yb, alpha)
            logZ_scale = scale_logZ * logZ
            dm_scale = scale_logZ * dm
            dv_scale = scale_logZ * dv
            sgp_grad_hyper, dx = sgp_layer.backprop_grads_lvm_mc(
                m_s, v_s, dm_scale, dv_scale, kfu_s, x_s, alpha)
            sgp_grad_input = sgp_layer.backprop_grads_reparam(
                dx, mx, vx, eps)
            lik_grad_hyper = lik_layer.backprop_grads(
                m, v, dm, dv, alpha, scale_logZ)
        else:
            raise NotImplementedError('propagation mode not implemented')

        grad_all = {}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute grad wrt x params
        dmx = sgp_grad_input['mx']
        dvx = sgp_grad_input['vx']

        x_KL_term, dkl_dmx, dkl_dvx = self.compute_KL_x(
            mx, vx, self.prior_mean, self.prior_var)
        scale_x = N * 1.0 / batch_size
        dmx += scale_x * dkl_dmx
        dvx += scale_x * dkl_dvx
        grad_all['x1'] = np.zeros_like(self.factor_x1)
        grad_all['x2'] = np.zeros_like(self.factor_x2)
        grad_all['x1'][idxs, :] = dmx / post_t2
        grad_all['x2'][idxs, :] = -dmx * post_t1 / post_t2**2 - dvx / post_t2**2
        grad_all['x2'][idxs, :] *= 2 * t12

        # compute objective
        sgp_KL_term = self.sgp_layer.compute_KL()
        x_KL_term = scale_x * x_KL_term
        energy = logZ_scale + x_KL_term + sgp_KL_term

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def compute_KL_x(self, mx, vx, m0, v0):
        kl = 0.5 * (np.log(v0) - np.log(vx) + (vx + (mx - m0)**2) / v0 - 1)
        kl_sum = np.sum(kl)
        dkl_dmx = (mx - m0) / v0
        dkl_dvx = - 0.5 / vx + 0.5 / v0
        return kl_sum, dkl_dmx, dkl_dvx

    def predict_f(self, inputs):
        """Summary

        Args:
            inputs (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not self.updated:
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf, _ = self.sgp_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def predict_y(self, inputs):
        """Summary

        Args:
            inputs (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not self.updated:
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def get_posterior_x(self):
        """Summary

        Returns:
            TYPE: Description
        """
        post_1 = self.prior_x1 + self.factor_x1
        post_2 = self.prior_x2 + self.factor_x2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def impute_missing(self, y, missing_mask, alpha=0.5, no_iters=10, add_noise=False):
        """Summary

        Args:
            y (TYPE): Description
            missing_mask (TYPE): Description
            alpha (float, optional): Description
            no_iters (int, optional): Description
            add_noise (bool, optional): Description

        Returns:
            TYPE: Description
        """
        # TODO
        # find latent conditioned on observed variables
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True

        N_test = y.shape[0]
        Q = self.Din
        # np.zeros((N_test, Q))
        factor_x1 = np.zeros((N_test, Q))
        factor_x2 = np.zeros((N_test, Q))
        for i in range(no_iters):
            # compute cavity of x
            cav_x1 = self.prior_x1 + (1 - alpha) * factor_x1
            cav_x2 = self.prior_x2 + (1 - alpha) * factor_x2
            cav_m = cav_x1 / cav_x2
            cav_v = 1.0 / cav_x2

            # propagate x through posterior and get gradients wrt mx and vx
            logZ, m, v, dlogZ_dmx, dlogZ_dvx = \
                self.sgp_layer.compute_logZ_and_gradients_imputation(
                    cav_m, cav_v,
                    y, missing_mask, alpha=alpha)

            # compute new posterior
            new_m = cav_m + cav_v * dlogZ_dmx
            new_v = cav_v - cav_v**2 * (dlogZ_dmx**2 - 2 * dlogZ_dvx)

            new_x2 = 1.0 / new_v
            new_x1 = new_x2 * new_m
            frac_x1 = new_x1 - cav_x1
            frac_x2 = new_x2 - cav_x2

            # update factor
            factor_x1 = (1 - alpha) * factor_x1 + frac_x1
            factor_x2 = (1 - alpha) * factor_x2 + frac_x2

        # compute posterior of x
        post_x1 = self.prior_x1 + factor_x1
        post_x2 = self.prior_x2 + factor_x2
        post_m = post_x1 / post_x2
        post_v = 1.0 / post_x2

        # propagate x forward to predict missing points
        my, vy = self.sgp_layer.forward_prop_thru_post(
            post_m, post_v, add_noise=add_noise)

        return my, vy

    def init_hypers_old(self, y_train):
        """Summary

        Args:
            y_train (TYPE): Description

        Returns:
            TYPE: Description
        """
        sgp_params = self.sgp_layer.init_hypers()
        lik_params = self.lik_layer.init_hypers()
        # TODO: alternatitve method for non real-valued data
        post_m = PCA_reduce(y_train, self.Din)
        post_m_mean = np.mean(post_m, axis=0)
        post_m_std = np.std(post_m, axis=0)
        post_m = (post_m - post_m_mean) / post_m_std
        post_v = 0.1 * np.ones_like(post_m)
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        x_params = {}
        x_params['x1'] = post_1
        x_params['x2'] = np.log(post_2 - 1) / 2

        init_params = dict(sgp_params)
        init_params.update(lik_params)
        init_params.update(x_params)
        return init_params

    def init_hypers(self, y_train):
        """Summary

        Args:
            y_train (TYPE): Description

        Returns:
            TYPE: Description
        """
        # TODO: alternatitve method for non real-valued data
        post_m = PCA_reduce(y_train, self.Din)
        post_m_mean = np.mean(post_m, axis=0)
        post_m_std = np.std(post_m, axis=0)
        post_m = (post_m - post_m_mean) / post_m_std
        post_v = 0.1 * np.ones_like(post_m)
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        x_params = {}
        x_params['x1'] = post_1
        x_params['x2'] = np.log(post_2 - 1) / 2
        # learnt a GP mapping between hidden states
        print 'init latent function using GPR...'
        x = post_m
        y = y_train
        reg = SGPR(x, y, self.M, 'Gaussian')
        reg.set_fixed_params(['sn', 'sf', 'ls', 'zu'])
        reg.optimise(method='L-BFGS-B', maxiter=100, disp=False)
        sgp_params = reg.sgp_layer.get_hypers()
        lik_params = self.lik_layer.init_hypers()
        init_params = dict(sgp_params)
        init_params.update(lik_params)
        init_params.update(x_params)
        return init_params

    def get_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        sgp_params = self.sgp_layer.get_hypers()
        lik_params = self.lik_layer.get_hypers()
        x_params = {}
        x_params['x1'] = self.factor_x1
        x_params['x2'] = np.log(self.factor_x2) / 2.0

        params = dict(sgp_params)
        params.update(lik_params)
        params.update(x_params)
        return params

    def update_hypers(self, params):
        """Summary

        Args:
            params (TYPE): Description
        """
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)
        self.factor_x1 = params['x1']
        self.factor_x2 = np.exp(2 * params['x2'])


