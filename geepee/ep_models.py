import sys
import math
import numpy as np
import scipy.linalg as npalg
from scipy import special
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import pdb
from scipy.cluster.vq import kmeans2

from utils import *
from kernels import *


# ideally these should be moved to some config file
jitter = 1e-6
gh_degree = 10


class SGP_Layer(object):

    def __init__(self, no_train, input_size, output_size, no_pseudo):
        self.Din = Din = input_size
        self.Dout = Dout = output_size
        self.M = M = no_pseudo
        self.N = N = no_train

        # factor variables
        self.t1 = np.zeros([N, Dout, M])
        self.t2 = np.zeros([N, Dout, M, M])
        
        # TODO
        self.mu = np.zeros([Dout, M, ])
        self.Su = np.zeros([Dout, M, M])
        self.SuinvMu = np.zeros([Dout, M, ])
        self.Suinv = np.zeros([Dout, M, M])
        self.Splusmm = np.zeros([Dout, M, M])

        # numpy variable for inducing points, Kuuinv, Kuu and its gradients
        self.zu = np.zeros([M, Din])
        self.Kuu = np.zeros([M, M])
        self.Kuuinv = np.zeros([M, M])

        # variables for the hyperparameters
        self.ls = np.zeros([Din, ])
        self.sf = 0

    def compute_phi_prior(self):
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        logZ_prior = self.Dout * 0.5 * logdet
        return logZ_prior

    def compute_phi_posterior(self):
        (sign, logdet) = np.linalg.slogdet(self.Su)
        phi_posterior = 0.5 * np.sum(logdet)
        phi_posterior += 0.5 * np.sum(self.mu*np.linalg.solve(self.Su, self.mu))
        return phi_posterior

    def compute_phi_cavity(self):
        logZ_posterior = 0
        (sign, logdet) = np.linalg.slogdet(self.Suhat)
        phi_cavity = 0.5 * np.sum(logdet)
        phi_cavity += 0.5 * np.sum(self.muhat*np.linalg.solve(self.Suhat, self.muhat))
        return phi_cavity

    def compute_phi(self, alpha=1.0):
        N = self.N
        scale_post = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        phi_prior = self.compute_phi_prior()
        phi_post = self.compute_phi_posterior()
        phi_cav = self.compute_phi_cavity()
        phi = scale_prior*phi_prior + scale_post*phi_post + scale_cav*phi_cav
        return phi

    def forward_prop_thru_cav(self, n, mx, vx=None, alpha=1.0):
        if vx is None:
            return self._forward_prop_deterministic_thru_cav(n, mx, alpha)
        else:
            return self._forward_prop_random_thru_cav_mm(n, mx, vx, alpha)

    def _forward_prop_deterministic_thru_cav(self, n, x, alpha):
        muhat, Suhat, SuinvMuhat, Suinvhat = self.compute_cavity(n, alpha)
        Kuuinv = self.Kuuinv
        Ahat = np.einsum('ab,ndb->nda', Kuuinv, muhat)
        Bhat = np.einsum(
            'ab,ndbc->ndac', 
            Kuuinv, np.einsum('ndab,bc->ndac', Suhat, Kuuinv)) - Kuuinv
        kff = np.exp(2*self.sf)
        kfu = compute_kernel(2*self.ls, 2*self.sf, x, self.zu)
        mout = np.einsum('nm,ndm->nd', kfu, Ahat)
        Bkfukuf = np.einsum('ndab,na,nb->nd', Bhat, kfu, kfu)
        vout = kff + Bkfukuf
        extra_res = [muhat, Suhat, SuinvMuhat, Suinvhat, kfu, Ahat, Bhat]
        return mout, vout, extra_res

    def _forward_prop_random_thru_cav_mm(self, n, mx, vx, alpha):
        muhat, Suhat, SuinvMuhat, Suinvhat = self.compute_cavity(n, alpha)
        Kuuinv = self.Kuuinv
        Ahat = np.einsum('ab,ndb->nda', Kuuinv, muhat)
        Smm = Suhat + np.einsum('nda,ndb->ndab', muhat, muhat)
        Bhat = np.einsum(
            'ab,ndbc->ndac', 
            Kuuinv, np.einsum('ndab,bc->ndac', Smm, Kuuinv)) - Kuuinv
        psi0 = np.exp(2*self.sf)
        psi1, psi2 = compute_psi_weave(2*self.ls, 2*self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,ndm->nd', psi1, Ahat)
        Bhatpsi2 = np.einsum('ndab,nab->nd', Bhat, psi2)
        vout = psi0 + Bhatpsi2 - mout**2
        extra_res = [muhat, Suhat, SuinvMuhat, Suinvhat, Smm, psi1, psi2, Ahat, Bhat]
        if len(np.where(vout<0)[0]) > 0:
            pdb.set_trace()
        return mout, vout, extra_res

    def forward_prop_thru_post(self, mx, vx=None):
        if vx is None:
            return self._forward_prop_deterministic_thru_post(mx)
        else:
            return self._forward_prop_random_thru_post_mm(mx, vx)

    def _forward_prop_deterministic_thru_post(self, x):
        Kuuinv = self.Kuuinv
        A = np.einsum('ab,db->da', Kuuinv, self.mu)
        B = np.einsum(
            'ab,dbc->dac', 
            Kuuinv, np.einsum('dab,bc->dac', self.Su, Kuuinv)) - Kuuinv
        kff = np.exp(2*self.sf)
        kfu = compute_kernel(2*self.ls, 2*self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', kfu, A)
        Bpsi2 = np.einsum('dab,na,nb->nd', B, kfu, kfu)
        vout = kff + Bpsi2
        return mout, vout

    # TODO
    def _forward_prop_random_thru_post_mm(self, mx, vx):
        Kuuinv = self.Kuuinv
        A = np.einsum('ab,db->da', Kuuinv, self.mu)
        Smm = self.Su + np.einsum('da,db->dab', self.mu, self.mu)
        B = np.einsum(
            'ab,dbc->dac', 
            Kuuinv, np.einsum('dab,bc->dac', Smm, Kuuinv)) - Kuuinv
        psi0 = np.exp(2.0*self.sf)
        psi1, psi2 = compute_psi_weave(2*self.ls, 2*self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, A)
        Bpsi2 = np.einsum('dab,nab->nd', B, psi2)
        vout = psi0 + Bpsi2 - mout**2
        return mout, vout

    def backprop_grads_lvm(self, m, v, dm, dv, extra_args, mx, vx, alpha=1.0):
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2*self.sf)
        zu = self.zu
        Kuuinv = self.Kuuinv
        a = extra_args
        muhat, Suhat, SuinvMuhat, Suinvhat, Smm, psi1, psi2, Ahat, Bhat = \
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]

        # compute grads wrt Ahat and Bhat
        dm_all = dm - 2 * dv * m
        dAhat = np.einsum('nd,nm->ndm', dm_all, psi1)
        dBhat = np.einsum('nd,nab->ndab', dv, psi2)
        # compute grads wrt psi1 and psi2
        dpsi1 = np.einsum('nd,ndm->nm', dm_all, Ahat)
        dpsi2 = np.einsum('nd,ndab->nab', dv, Bhat)
        dsf2, dls, dzu, dmx, dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, ls, sf2, mx, vx, zu)

        dvcav = np.einsum('ab,ndbc,ce->ndae', Kuuinv, dBhat, Kuuinv)
        dmcav = 2 * np.einsum('ndab,ndb->nda', dvcav, muhat) \
            + np.einsum('ab,ndb->nda', Kuuinv, dAhat)

        grad_hyper = {} 
        grad_input = {'mx': dmx, 'vx': dvx, 'mcav': dmcav, 'vcav': dvcav}

        return grad_hyper, grad_input

    def backprop_grads_reg(self, m, v, dm, dv, extra_args, x, alpha=1.0):
        a = extra_args
        muhat, Suhat, SuinvMuhat, Suinvhat, kfu, Ahat, Bhat = \
            a[0], a[1], a[2], a[3], a[4], a[5], a[6]
        Kuuinv = self.Kuuinv
        # compute grads wrt Ahat and Bhat
        dAhat = np.einsum('nd,nm->ndm', dm, kfu)
        dBhat = np.einsum('nd,na,nb->ndab', dv, kfu, kfu)

        dvcav = np.einsum('ab,ndbc,ce->ndae', Kuuinv, dBhat, Kuuinv)
        dmcav = np.einsum('ab,ndb->nda', Kuuinv, dAhat)

        grad_hyper = {}
        grad_cav = {'mcav': dmcav, 'vcav': dvcav}

        return grad_hyper, grad_cav

    def update_factor(self, n, alpha, grad_cav, extra_args, decay=0):
        muhat, Suhat, SuinvMuhat, Suinvhat = \
            extra_args[0], extra_args[1], extra_args[2], extra_args[3]
        dmcav, dvcav = grad_cav['mcav'], grad_cav['vcav']

        # perform Power-EP update
        munew = muhat + np.einsum('ndab,ndb->nda', Suhat, dmcav)
        inner = np.einsum('nda,ndb->ndab', dmcav, dmcav) - 2*dvcav
        Sunew = Suhat - np.einsum(
            'ndab,ndbc->ndac', 
            Suhat, np.einsum('ndab,ndbc->ndac', inner, Suhat))
        Suinvnew = np.linalg.inv(Sunew)
        SuinvMunew = np.einsum('ndab,ndb->nda', Suinvnew, munew)
        t2_frac = Suinvnew - Suinvhat
        t1_frac = SuinvMunew - SuinvMuhat
        t1_old = self.t1[n, :, :]
        t2_old = self.t2[n, :, :, :]
        t1_new = (1.0-alpha) * t1_old + t1_frac
        t2_new = (1.0-alpha) * t2_old + t2_frac

        if t1_new.shape[0] == 1:
            # TODO: do damping here?
            self.t1[n, :, :] = t1_new
            self.t2[n, :, :, :] = t2_new
            # TODO: update posterior
            self.Su = Sunew
            self.mu = munew
            self.Suinv = Suinvnew
            self.SuinvMu = SuinvMunew
        else:
            # parallel update
            self.t1[n, :, :] = decay * t1_old + (1-decay) * t1_new
            self.t2[n, :, :] = decay * t2_old + (1-decay) * t2_new
            self.update_posterior()

    def sample(self, x):
        Su = self.Su
        mu = self.mu
        Lu = np.linalg.cholesky(Su)
        epsilon = np.random.randn(self.Dout, self.M)
        u_sample = mu + np.einsum('dab,db->da', Lu, epsilon)

        kff = compute_kernel(2*self.ls, 2*self.sf, x, x) 
        kff += np.diag(jitter * np.ones(x.shape[0]))
        kfu = compute_kernel(2*self.ls, 2*self.sf, x, self.zu)
        qfu = np.dot(kfu, self.Kuuinv)
        mf = np.einsum('nm,dm->nd', qfu, u_sample)
        vf = kff - np.dot(qfu, kfu.T)
        Lf = np.linalg.cholesky(vf)
        epsilon = np.random.randn(x.shape[0], self.Dout)
        f_sample = mf + np.einsum('ab,bd->ad', Lf, epsilon)
        return f_sample

    def compute_kuu(self):
        # update kuu and kuuinv
        ls = self.ls
        sf = self.sf
        Dout = self.Dout
        M = self.M
        zu = self.zu
        self.Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        self.Kuu += np.diag(jitter * np.ones((M, )))
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def compute_cavity(self, n, alpha=1.0):
        # compute the leave one out moments
        t1n = self.t1[n, :, :]
        t2n = self.t2[n, :, :, :]
        Suinvhat = self.Suinv - alpha*t2n
        SuinvMuhat = self.SuinvMu - alpha*t1n
        Suhat = np.linalg.inv(Suinvhat)
        muhat = np.einsum('ndab,ndb->nda', Suhat, SuinvMuhat)
        return muhat, Suhat, SuinvMuhat, Suinvhat

    def update_posterior(self):
        # compute the posterior approximation
        self.Suinv = self.Kuuinv + np.sum(self.t2, axis=0)
        self.SuinvMu = np.sum(self.t1, axis=0)
        self.Su = np.linalg.inv(self.Suinv)
        self.mu = np.einsum('dab,db->da', self.Su, self.SuinvMu)

    def init_hypers(self, x_train=None, key_suffix=''):
        # dict to hold hypers, inducing points and parameters of q(U)
        N = self.N
        M = self.M
        Din = self.Din
        Dout = self.Dout

        if x_train is None:
            ls = np.log(np.ones((Din, )) + 0.1 * np.random.rand(Din, ))
            sf = np.log(np.array([1]))
            zu = np.tile(np.linspace(-1, 1, M).reshape((M, 1)), (1, Din))
        else:
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
                ls[i] = np.log(d2imed  + 1e-16)
            sf = np.log(np.array([0.5]))
        
        params = dict()
        params['sf' + key_suffix] = sf
        params['ls' + key_suffix] = ls
        params['zu' + key_suffix] = zu

        return params

    def get_hypers(self, key_suffix=''):
        params = {}
        M = self.M
        Din = self.Din
        Dout = self.Dout
        params['ls' + key_suffix] = self.ls
        params['sf' + key_suffix] = self.sf
        params_zu_i = self.zu
        params['zu' + key_suffix] = self.zu
        return params

    def update_hypers(self, params, key_suffix=''):
        self.ls = params['ls' + key_suffix]
        self.sf = params['sf' + key_suffix]
        self.zu = params['zu' + key_suffix]

        # update Kuu given new hypers
        self.compute_kuu()
        # compute mu and Su for each layer
        self.update_posterior()


class Lik_Layer(object):
    def __init__(self, N, D):
        self.N = N
        self.D = D

    def compute_log_Z(self, mout, vout, y, alpha=1.0):
        pass

    def backprop_grads(self, mout, vout, dmout, dvout, alpha=1.0, scale=1.0):
        return {}

    def init_hypers(self):
        return {}

    def get_hypers(self):
        return {}

    def update_hypers(self, params):
        pass


class Gauss_Layer(Lik_Layer):
    
    def __init__(self, N, D):
        super(Gauss_Layer, self).__init__(N, D)
        self.sn = 0
    
    def compute_log_Z(self, mout, vout, y, alpha=1.0):
        # real valued data, gaussian lik
        sn2 = np.exp(2.0 * self.sn)
        vout += sn2 / alpha
        if len(np.where(vout<0)[0]) > 0:
            pdb.set_trace()
        logZ = np.sum(-0.5 * (np.log(2 * np.pi * vout) +
                              (y - mout)**2 / vout))
        logZ += y.shape[0] * self.D * (0.5 * np.log(2 * np.pi * sn2 / alpha)
                            - 0.5 * alpha * np.log(2 * np.pi * sn2))
        dlogZ_dm = (y - mout) / vout
        dlogZ_dv = -0.5 / vout + 0.5 * (y - mout)**2 / vout**2

        return logZ, dlogZ_dm, dlogZ_dv

    def backprop_grads(self, mout, vout, dmout, dvout, alpha=1.0, scale=1.0):
        sn2 = np.exp(2.0 * self.sn)
        dv_sum = np.sum(dvout)
        dsn = dv_sum*2*sn2/alpha + mout.shape[0]*self.D*(1-alpha)
        dsn *= scale
        return {'sn': dsn}

    def init_hypers(self):
        self.sn = np.log(0.01)
        return {'sn': self.sn}

    def get_hypers(self):
        return {'sn': self.sn}    

    def update_hypers(self, params):
        self.sn = params['sn']


class Probit_Layer(Lik_Layer):

    __gh_points = None
    def _gh_points(self, T=20):
        if self.__gh_points is None:
            self.__gh_points = np.polynomial.hermite.hermgauss(T)
        return self.__gh_points
    
    def compute_log_Z(self, mout, vout, y, alpha=1.0):
        # binary data probit likelihood
        if alpha == 1.0:
            t = y * mout / np.sqrt(1 + vout)
            Z = 0.5 * (1 + special.erf(t / np.sqrt(2)))
            eps = 1e-16
            logZ = np.sum(np.log(Z + eps))

            dlogZ_dt = 1 / (Z + eps) * 1 / np.sqrt(2 *
                                                   np.pi) * np.exp(-t**2.0 / 2)
            dt_dm = y / np.sqrt(1 + vout)
            dt_dv = -0.5 * y * mout / (1 + vout)**1.5
            dlogZ_dm = dlogZ_dt * dt_dm
            dlogZ_dv = dlogZ_dt * dt_dv

        else:
            gh_x, gh_w = self._gh_points(gh_degree)
            gh_x = gh_x[:, np.newaxis, np.newaxis]
            gh_w = gh_w[:, np.newaxis, np.newaxis]

            ts = gh_x * np.sqrt(2*vout[np.newaxis, :, :]) + mout[np.newaxis, :, :]
            eps = 1e-8
            pdfs = 0.5 * (1 + special.erf(y*ts / np.sqrt(2))) + eps
            Ztilted = np.sum(pdfs**alpha * gh_w, axis=0) / np.sqrt(np.pi)
            logZ = np.sum(np.log(Ztilted))
            
            a = pdfs**(alpha-1.0)*np.exp(-ts**2/2)
            dZdm = np.sum(gh_w * a, axis=0) * y * alpha / np.pi / np.sqrt(2)
            dlogZ_dm = dZdm / Ztilted + eps

            dZdv = np.sum(gh_w * (a*gh_x), axis=0) * y * alpha / np.pi / np.sqrt(2) / np.sqrt(2*vout)
            dlogZ_dv = dZdv / Ztilted + eps

        return logZ, dlogZ_dm, dlogZ_dv


class EP_Model(object):

    def __init__(self, y_train):
        self.y_train = y_train
        self.N = y_train.shape[0]
        self.fixed_params = []
        self.updated = False

    def init_hypers(self, y_train=None, x_train=None):
        pass

    def get_hypers(self):
        pass

    def inference(self, alpha, no_epochs=10):
        pass

    def optimise(
        self, method='L-BFGS-B', tol=None, reinit_hypers=True, 
        callback=None, maxiter=1000, alpha=0.5, adam_lr=0.05, **kargs):
        self.updated = False

        if reinit_hypers:
            init_params_dict = self.init_hypers(self.y_train)
        else:
            init_params_dict = self.get_hypers()

        init_params_vec, params_args = flatten_dict(init_params_dict)

        N = self.N
        idxs = np.arange(N)

        try:
            if method.lower() == 'adam':
                results = adam(objective_wrapper, init_params_vec,
                               step_size=adam_lr,
                               maxiter=maxiter,
                               args=(params_args, self, idxs, alpha))
            else:
                options = {'maxiter': maxiter, 'disp': True, 'gtol': 1e-8}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, idxs, alpha),
                    method=method,
                    jac=True,
                    tol=tol,
                    callback=callback,
                    options=options)

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'
            results = []
            # todo: deal with rresults here

        results = self.get_hypers()
        return results

    def set_fixed_params(self, params):
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)


class SGPR(EP_Model):

    def __init__(self, x_train, y_train, no_pseudo, lik='Gaussian'):
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

    def inference(self, alpha=1.0, no_epochs=10, parallel=False, decay=0.5):
        try:
            for e in range(no_epochs):
                print 'epoch %d/%d' % (e, no_epochs)
                if not parallel:
                    for n in range(self.N):
                        yn = self.y_train[n, :].reshape([1, self.Dout])
                        xn = self.x_train[n, :].reshape([1, self.Din])
                        (mn, vn, extra_res) = \
                            self.sgp_layer.forward_prop_thru_cav([n], xn, alpha=alpha)
                        logZn, dmn, dvn = \
                            self.lik_layer.compute_log_Z(mn, vn, yn, alpha)
                        grad_hyper, grad_cav = self.sgp_layer.backprop_grads_reg(
                            mn, vn, dmn, dvn, extra_res, xn, alpha=alpha)
                        self.sgp_layer.update_factor([n], alpha, grad_cav, extra_res)
                else:
                    # parallel update for entire dataset
                    # TODO: minibatch parallel
                    idxs = np.arange(self.N)
                    y = self.y_train[idxs, :]
                    x = self.x_train[idxs, :]
                    (m, v, extra_res) = \
                        self.sgp_layer.forward_prop_thru_cav(idxs, x, alpha=alpha)
                    logZ, dm, dv = \
                        self.lik_layer.compute_log_Z(m, v, y, alpha)
                    grad_hyper, grad_cav = self.sgp_layer.backprop_grads_reg(
                        m, v, dm, dv, extra_res, x, alpha=alpha)
                    self.sgp_layer.update_factor(
                        idxs, alpha, grad_cav, extra_res, decay=decay)

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'

    def predict_f(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def sample_f(self, inputs, no_samples=1):
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
        if not self.updated:
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def update_hypers(self, params):
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)

    def init_hypers(self):
        sgp_params = self.sgp_layer.init_hypers(self.x_train)
        lik_params = self.lik_layer.init_hypers()
        init_params = dict(sgp_params)
        init_params.update(lik_params)
        return init_params

    def get_hypers(self):
        sgp_params = self.sgp_layer.get_hypers()
        lik_params = self.lik_layer.get_hypers()
        params = dict(sgp_params)
        params.update(lik_params)
        return params


class SGPLVM(EP_Model):

    def __init__(self, y_train, hidden_size, no_pseudo, 
        lik='Gaussian', prior_mean=0, prior_var=1):
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
        self.tx1 = np.zeros((N, Din))
        self.tx2 = np.zeros((N, Din))

        self.t01 = prior_mean / prior_var
        self.t02 = 1.0 / prior_var

    def inference(self, alpha=1.0, no_epochs=10, parallel=False, decay=0):
        try:
            for e in range(no_epochs):
                print 'epoch %d/%d' % (e, no_epochs)
                if not parallel:
                    for n in range(self.N):
                        yn = self.y_train[n, :].reshape([1, self.Dout])
                        cav_m_n, cav_v_n, _, _ = self.compute_cavity_x([n], alpha)
                        (mn, vn, extra_res) = \
                            self.sgp_layer.forward_prop_thru_cav([n], cav_m_n, cav_v_n, alpha=alpha)
                        logZn, dmn, dvn = \
                            self.lik_layer.compute_log_Z(mn, vn, yn, alpha)
                        grad_hyper, grad_cav = self.sgp_layer.backprop_grads_lvm(
                            mn, vn, dmn, dvn, extra_res, cav_m_n, cav_v_n, alpha=alpha)
                        self.sgp_layer.update_factor([n], alpha, grad_cav, extra_res)
                        self.update_factor_x([n], alpha, grad_cav, cav_m_n, cav_v_n, decay=decay)
                else:
                    # parallel update for entire dataset
                    # TODO: minibatch parallel
                    idxs = np.arange(self.N)
                    y = self.y_train[idxs, :]
                    cav_m, cav_v, _, _ = self.compute_cavity_x(idxs, alpha)
                    (m, v, extra_res) = \
                        self.sgp_layer.forward_prop_thru_cav(idxs, cav_m, cav_v, alpha=alpha)
                    logZ, dm, dv = \
                        self.lik_layer.compute_log_Z(m, v, y, alpha)
                    grad_hyper, grad_cav = self.sgp_layer.backprop_grads_lvm(
                        m, v, dm, dv, extra_res, cav_m, cav_v, alpha=alpha)
                    self.sgp_layer.update_factor(idxs, alpha, grad_cav, extra_res, decay=decay)
                    # self.update_factor_x(idxs, alpha, grad_cav, cav_m, cav_v, decay=decay)
                    dmx = grad_cav['mx']
                    dvx = grad_cav['vx']
                    new_m = cav_m + cav_v * dmx
                    new_v = cav_v - cav_v**2 * (dmx**2 - 2*dvx)
                    new_p2 = 1.0 / new_v
                    new_p1 = new_p2 * new_m

                    frac_t2 = new_p2 - 1.0 / cav_v
                    frac_t1 = new_p1 - cav_m / cav_v
                    cur_t1 = self.tx1[idxs, :]
                    cur_t2 = self.tx2[idxs, :]
                    tx1_new = (1-alpha) * cur_t1 + frac_t1
                    tx2_new = (1-alpha) * cur_t2 + frac_t2
                    tx1_new = decay * cur_t1 + (1-decay) * tx1_new
                    tx2_new = decay * cur_t2 + (1-decay) * tx2_new

                    neg_idxs = np.where(np.abs(tx2_new) < 1e-6) and np.where(tx2_new < 0)
                    tx2_new[neg_idxs] = 0
                    print neg_idxs
                    # if len(neg_idxs[0]) > 0:
                    #     pdb.set_trace()
                    # tx2_new[neg_idxs] = cur_t2[neg_idxs]
                    # tx1_new[neg_idxs] = cur_t1[neg_idxs]

                    self.tx1[idxs, :] = tx1_new
                    self.tx2[idxs, :] = tx2_new

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'

    def compute_cavity_x(self, n, alpha):
        # prior factor
        cav_x1 = self.t01 + (1-alpha) * self.tx1[n, :]
        cav_x2 = self.t02 + (1-alpha) * self.tx2[n, :]
        cav_v = 1.0 / cav_x2
        cav_m = cav_v * cav_x1
        return cav_m, cav_v, cav_x1, cav_x2

    def update_factor_x(self, n, alpha, grad_cav, cav_m, cav_v, decay=0.0):
        dmx = grad_cav['mx']
        dvx = grad_cav['vx']
        new_m = cav_m + cav_v * dmx
        new_v = cav_v - cav_v**2 * (dmx**2 - 2*dvx)
        new_p2 = 1.0 / new_v
        new_p1 = new_p2 * new_m

        frac_t2 = new_p2 - 1.0 / cav_v
        frac_t1 = new_p1 - cav_m / cav_v
        cur_t1 = self.tx1[n, :]
        cur_t2 = self.tx2[n, :]
        tx1_new = (1-alpha) * cur_t1 + frac_t1
        tx2_new = (1-alpha) * cur_t2 + frac_t2
        tx1_new = decay * cur_t1 + (1-decay) * tx1_new
        tx2_new = decay * cur_t2 + (1-decay) * tx2_new

        neg_idxs = np.where(tx2_new < 0)
        print neg_idxs
        tx2_new[neg_idxs] = cur_t2[neg_idxs]
        tx1_new[neg_idxs] = cur_t1[neg_idxs]

        self.tx1[n, :] = tx1_new
        self.tx2[n, :] = tx2_new

    def get_posterior_x(self):
        post_1 = self.t01 + self.tx1
        post_2 = self.t02 + self.tx2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def predict_f(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def sample_f(self, inputs, no_samples=1):
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
        if not self.updated:
            self.sgp_layer.update_posterior()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def update_hypers(self, params):
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)

    def init_hypers(self):
        sgp_params = self.sgp_layer.init_hypers(self.x_train)
        lik_params = self.lik_layer.init_hypers()
        init_params = dict(sgp_params)
        init_params.update(lik_params)
        return init_params

    def get_hypers(self):
        sgp_params = self.sgp_layer.get_hypers()
        lik_params = self.lik_layer.get_hypers()
        params = dict(sgp_params)
        params.update(lik_params)
        return params
