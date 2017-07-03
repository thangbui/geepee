"""Summary
# TODO: this should reuse base models!
"""
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
from lik_layers import Gauss_Layer, Probit_Layer, Gauss_Emis
from base_models import Base_SGPR, Base_SGPLVM, Base_SGPSSM
from config import *


class SGP_Layer(object):
    """Summary

    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        Kuu (TYPE): Description
        Kuuinv (TYPE): Description
        ls (TYPE): Description
        M (TYPE): Description
        mu (TYPE): Description
        N (TYPE): Description
        sf (int): Description
        Splusmm (TYPE): Description
        Su (TYPE): Description
        Suinv (TYPE): Description
        SuinvMu (TYPE): Description
        t1 (TYPE): Description
        t2 (TYPE): Description
        zu (TYPE): Description
    """

    def __init__(self, no_train, input_size, output_size, no_pseudo):
        """Summary

        Args:
            no_train (TYPE): Description
            input_size (TYPE): Description
            output_size (TYPE): Description
            no_pseudo (TYPE): Description
        """
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
        """Summary

        Returns:
            TYPE: Description
        """
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        logZ_prior = self.Dout * 0.5 * logdet
        return logZ_prior

    def compute_phi_posterior(self):
        """Summary

        Returns:
            TYPE: Description
        """
        (sign, logdet) = np.linalg.slogdet(self.Su)
        phi_posterior = 0.5 * np.sum(logdet)
        phi_posterior += 0.5 * \
            np.sum(self.mu * np.linalg.solve(self.Su, self.mu))
        return phi_posterior

    def compute_phi_cavity(self):
        """Summary

        Returns:
            TYPE: Description
        """
        logZ_posterior = 0
        (sign, logdet) = np.linalg.slogdet(self.Suhat)
        phi_cavity = 0.5 * np.sum(logdet)
        phi_cavity += 0.5 * \
            np.sum(self.muhat * np.linalg.solve(self.Suhat, self.muhat))
        return phi_cavity

    def compute_phi(self, alpha=1.0):
        """Summary

        Args:
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
        N = self.N
        scale_post = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        phi_prior = self.compute_phi_prior()
        phi_post = self.compute_phi_posterior()
        phi_cav = self.compute_phi_cavity()
        phi = scale_prior * phi_prior + scale_post * phi_post + scale_cav * phi_cav
        return phi

    def forward_prop_thru_cav(self, n, mx, vx=None, alpha=1.0):
        """Summary

        Args:
            n (TYPE): Description
            mx (TYPE): Description
            vx (None, optional): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
        if vx is None:
            return self._forward_prop_deterministic_thru_cav(n, mx, alpha)
        else:
            return self._forward_prop_random_thru_cav_mm(n, mx, vx, alpha)

    def _forward_prop_deterministic_thru_cav(self, n, x, alpha):
        """Summary

        Args:
            n (TYPE): Description
            x (TYPE): Description
            alpha (TYPE): Description

        Returns:
            TYPE: Description
        """
        muhat, Suhat, SuinvMuhat, Suinvhat = self.compute_cavity(n, alpha)
        Kuuinv = self.Kuuinv
        Ahat = np.einsum('ab,ndb->nda', Kuuinv, muhat)
        Bhat = np.einsum(
            'ab,ndbc->ndac',
            Kuuinv, np.einsum('ndab,bc->ndac', Suhat, Kuuinv)) - Kuuinv
        kff = np.exp(2 * self.sf)
        kfu = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        mout = np.einsum('nm,ndm->nd', kfu, Ahat)
        Bkfukuf = np.einsum('ndab,na,nb->nd', Bhat, kfu, kfu)
        vout = kff + Bkfukuf
        extra_res = [muhat, Suhat, SuinvMuhat, Suinvhat, kfu, Ahat, Bhat]
        return mout, vout, extra_res

    def _forward_prop_random_thru_cav_mm(self, n, mx, vx, alpha):
        """Summary

        Args:
            n (TYPE): Description
            mx (TYPE): Description
            vx (TYPE): Description
            alpha (TYPE): Description

        Returns:
            TYPE: Description
        """
        muhat, Suhat, SuinvMuhat, Suinvhat = self.compute_cavity(n, alpha)
        Kuuinv = self.Kuuinv
        Ahat = np.einsum('ab,ndb->nda', Kuuinv, muhat)
        Smm = Suhat + np.einsum('nda,ndb->ndab', muhat, muhat)
        Bhat = np.einsum(
            'ab,ndbc->ndac',
            Kuuinv, np.einsum('ndab,bc->ndac', Smm, Kuuinv)) - Kuuinv
        psi0 = np.exp(2 * self.sf)
        psi1, psi2 = compute_psi_weave(
            2 * self.ls, 2 * self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,ndm->nd', psi1, Ahat)
        Bhatpsi2 = np.einsum('ndab,nab->nd', Bhat, psi2)
        vout = psi0 + Bhatpsi2 - mout**2
        extra_res = [muhat, Suhat, SuinvMuhat,
                     Suinvhat, Smm, psi1, psi2, Ahat, Bhat]
        return mout, vout, extra_res

    def forward_prop_thru_post(self, mx, vx=None):
        """Summary

        Args:
            mx (TYPE): Description
            vx (None, optional): Description

        Returns:
            TYPE: Description
        """
        if vx is None:
            return self._forward_prop_deterministic_thru_post(mx)
        else:
            return self._forward_prop_random_thru_post_mm(mx, vx)

    def _forward_prop_deterministic_thru_post(self, x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        Kuuinv = self.Kuuinv
        A = np.einsum('ab,db->da', Kuuinv, self.mu)
        B = np.einsum(
            'ab,dbc->dac',
            Kuuinv, np.einsum('dab,bc->dac', self.Su, Kuuinv)) - Kuuinv
        kff = np.exp(2 * self.sf)
        kfu = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', kfu, A)
        Bpsi2 = np.einsum('dab,na,nb->nd', B, kfu, kfu)
        vout = kff + Bpsi2
        return mout, vout

    # TODO
    def _forward_prop_random_thru_post_mm(self, mx, vx):
        """Summary

        Args:
            mx (TYPE): Description
            vx (TYPE): Description

        Returns:
            TYPE: Description
        """
        Kuuinv = self.Kuuinv
        A = np.einsum('ab,db->da', Kuuinv, self.mu)
        Smm = self.Su + np.einsum('da,db->dab', self.mu, self.mu)
        B = np.einsum(
            'ab,dbc->dac',
            Kuuinv, np.einsum('dab,bc->dac', Smm, Kuuinv)) - Kuuinv
        psi0 = np.exp(2.0 * self.sf)
        psi1, psi2 = compute_psi_weave(
            2 * self.ls, 2 * self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, A)
        Bpsi2 = np.einsum('dab,nab->nd', B, psi2)
        vout = psi0 + Bpsi2 - mout**2
        return mout, vout

    def backprop_grads_lvm(self, m, v, dm, dv, extra_args, mx, vx, alpha=1.0):
        """Summary

        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm (TYPE): Description
            dv (TYPE): Description
            extra_args (TYPE): Description
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
        """Summary

        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm (TYPE): Description
            dv (TYPE): Description
            extra_args (TYPE): Description
            x (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
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
        """Summary

        Args:
            n (TYPE): Description
            alpha (TYPE): Description
            grad_cav (TYPE): Description
            extra_args (TYPE): Description
            decay (int, optional): Description

        Returns:
            TYPE: Description
        """
        muhat, Suhat, SuinvMuhat, Suinvhat = \
            extra_args[0], extra_args[1], extra_args[2], extra_args[3]
        dmcav, dvcav = grad_cav['mcav'], grad_cav['vcav']

        # perform Power-EP update
        munew = muhat + np.einsum('ndab,ndb->nda', Suhat, dmcav)
        inner = np.einsum('nda,ndb->ndab', dmcav, dmcav) - 2 * dvcav
        Sunew = Suhat - np.einsum(
            'ndab,ndbc->ndac',
            Suhat, np.einsum('ndab,ndbc->ndac', inner, Suhat))
        Suinvnew = np.linalg.inv(Sunew)
        SuinvMunew = np.einsum('ndab,ndb->nda', Suinvnew, munew)
        t2_frac = Suinvnew - Suinvhat
        t1_frac = SuinvMunew - SuinvMuhat
        t1_old = self.t1[n, :, :]
        t2_old = self.t2[n, :, :, :]
        t1_new = (1.0 - alpha) * t1_old + t1_frac
        t2_new = (1.0 - alpha) * t2_old + t2_frac

        if t1_new.shape[0] == 1:
            # TODO: do damping here?
            self.t1[n, :, :] = t1_new
            self.t2[n, :, :, :] = t2_new
            # TODO: update posterior
            self.Su = Sunew[0, :, :, :]
            self.mu = munew[0, :, :]
            self.Suinv = Suinvnew[0, :, :, :]
            self.SuinvMu = SuinvMunew[0, :, :]
        else:
            # parallel update
            self.t1[n, :, :] = decay * t1_old + (1 - decay) * t1_new
            self.t2[n, :, :, :] = decay * t2_old + (1 - decay) * t2_new
            self.update_posterior()

        # axs[0].errorbar(self.zu[:, 0]+0.05, self.mu[0, :], fmt='+r', yerr=np.sqrt(np.diag(self.Su[0, :, :])))
        # axs[1].errorbar(self.zu[:, 0]+0.05, self.mu[1, :], fmt='+r', yerr=np.sqrt(np.diag(self.Su[1, :, :])))
        # axs[0].set_title('n = %d' % n[0])
        # plt.show()

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
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def compute_cavity(self, n, alpha=1.0):
        """Summary

        Args:
            n (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
        # compute the leave one out moments
        t1n = self.t1[n, :, :]
        t2n = self.t2[n, :, :, :]
        Suinvhat = self.Suinv - alpha * t2n
        SuinvMuhat = self.SuinvMu - alpha * t1n
        Suhat = np.linalg.inv(Suinvhat)
        muhat = np.einsum('ndab,ndb->nda', Suhat, SuinvMuhat)
        return muhat, Suhat, SuinvMuhat, Suinvhat

    def update_posterior(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # compute the posterior approximation
        self.Suinv = self.Kuuinv + np.sum(self.t2, axis=0)
        self.SuinvMu = np.sum(self.t1, axis=0)
        self.Su = np.linalg.inv(self.Suinv)
        self.mu = np.einsum('dab,db->da', self.Su, self.SuinvMu)

    def init_hypers(self, x_train=None, key_suffix=''):
        """Summary

        Args:
            x_train (None, optional): Description
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
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
                ls[i] = np.log(d2imed + 1e-16)
            sf = np.log(np.array([0.5]))

        params = dict()
        params['sf' + key_suffix] = sf
        params['ls' + key_suffix] = ls
        params['zu' + key_suffix] = zu

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
        params_zu_i = self.zu
        params['zu' + key_suffix] = self.zu
        return params

    def update_hypers(self, params, key_suffix=''):
        """Summary

        Args:
            params (TYPE): Description
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
        self.ls = params['ls' + key_suffix]
        self.sf = params['sf' + key_suffix]
        self.zu = params['zu' + key_suffix]

        # update Kuu given new hypers
        self.compute_kuu()
        # compute mu and Su for each layer
        self.update_posterior()


class SGPR(Base_SGPR):
    """Summary

    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        lik_layer (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        sgp_layer (TYPE): Description
        updated (bool): Description
        x_train (TYPE): Description
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
        super(SGPR, self).__init__(x_train, y_train, no_pseudo, lik)
        self.sgp_layer = SGP_Layer(self.N, self.Din, self.Dout, self.M)

    def inference(self, alpha=1.0, no_epochs=10, parallel=False, decay=0.5):
        """Summary

        Args:
            alpha (float, optional): Description
            no_epochs (int, optional): Description
            parallel (bool, optional): Description
            decay (float, optional): Description

        Returns:
            TYPE: Description
        """
        try:
            for e in range(no_epochs):
                if e % 50 == 0:
                    print 'epoch %d/%d' % (e, no_epochs)
                if not parallel:
                    for n in range(self.N):
                        yn = self.y_train[n, :].reshape([1, self.Dout])
                        xn = self.x_train[n, :].reshape([1, self.Din])
                        (mn, vn, extra_res) = \
                            self.sgp_layer.forward_prop_thru_cav(
                                [n], xn, alpha=alpha)
                        logZn, dmn, dvn = \
                            self.lik_layer.compute_log_Z(mn, vn, yn, alpha)
                        grad_hyper, grad_cav = self.sgp_layer.backprop_grads_reg(
                            mn, vn, dmn, dvn, extra_res, xn, alpha=alpha)
                        self.sgp_layer.update_factor(
                            [n], alpha, grad_cav, extra_res)
                else:
                    # parallel update for entire dataset
                    # TODO: minibatch parallel
                    idxs = np.arange(self.N)
                    y = self.y_train[idxs, :]
                    x = self.x_train[idxs, :]
                    (m, v, extra_res) = \
                        self.sgp_layer.forward_prop_thru_cav(
                            idxs, x, alpha=alpha)
                    logZ, dm, dv = \
                        self.lik_layer.compute_log_Z(m, v, y, alpha)
                    grad_hyper, grad_cav = self.sgp_layer.backprop_grads_reg(
                        m, v, dm, dv, extra_res, x, alpha=alpha)
                    self.sgp_layer.update_factor(
                        idxs, alpha, grad_cav, extra_res, decay=decay)

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'


class SGPLVM(Base_SGPLVM):
    """Summary

    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        lik_layer (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        sgp_layer (TYPE): Description
        t01 (TYPE): Description
        t02 (TYPE): Description
        tx1 (TYPE): Description
        tx2 (TYPE): Description
        updated (bool): Description
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
        super(SGPLVM, self).__init__(y_train, hidden_size, no_pseudo, lik, prior_mean, prior_var)
        self.sgp_layer = SGP_Layer(self.N, self.Din, self.Dout, self.M)

        # natural params for latent variables
        self.tx1 = np.zeros((self.N, self.Din))
        self.tx2 = np.zeros((self.N, self.Din))

        self.t01 = prior_mean / prior_var
        self.t02 = 1.0 / prior_var

        # TODO: alternatitve method for non real-valued data
        post_m = PCA_reduce(y_train, self.Din)
        post_m_mean = np.mean(post_m, axis=0)
        post_m_std = np.std(post_m, axis=0)
        post_m = (post_m - post_m_mean) / post_m_std
        post_v = 0.1 * np.ones_like(post_m)
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        self.tx1 = post_1 - self.t01
        self.tx2 = post_2 - self.t02

    def inference(self, alpha=1.0, no_epochs=10, parallel=False, decay=0):
        """Summary

        Args:
            alpha (float, optional): Description
            no_epochs (int, optional): Description
            parallel (bool, optional): Description
            decay (int, optional): Description

        Returns:
            TYPE: Description
        """
        try:
            for e in range(no_epochs):
                if e % 50 == 0:
                    print 'epoch %d/%d' % (e, no_epochs)
                if not parallel:
                    for n in range(self.N):
                        yn = self.y_train[n, :].reshape([1, self.Dout])
                        cav_m_n, cav_v_n, _, _ = self.compute_cavity_x([
                                                                       n], alpha)
                        (mn, vn, extra_res) = \
                            self.sgp_layer.forward_prop_thru_cav(
                                [n], cav_m_n, cav_v_n, alpha=alpha)
                        logZn, dmn, dvn = \
                            self.lik_layer.compute_log_Z(mn, vn, yn, alpha)
                        grad_hyper, grad_cav = self.sgp_layer.backprop_grads_lvm(
                            mn, vn, dmn, dvn, extra_res, cav_m_n, cav_v_n, alpha=alpha)
                        self.sgp_layer.update_factor(
                            [n], alpha, grad_cav, extra_res, decay=decay)
                        self.update_factor_x(
                            [n], alpha, grad_cav, cav_m_n, cav_v_n, decay=decay)
                else:
                    # parallel update for entire dataset
                    # TODO: minibatch parallel
                    idxs = np.arange(self.N)
                    y = self.y_train[idxs, :]
                    cav_m, cav_v, _, _ = self.compute_cavity_x(idxs, alpha)
                    (m, v, extra_res) = \
                        self.sgp_layer.forward_prop_thru_cav(
                            idxs, cav_m, cav_v, alpha=alpha)
                    logZ, dm, dv = \
                        self.lik_layer.compute_log_Z(m, v, y, alpha)
                    grad_hyper, grad_cav = self.sgp_layer.backprop_grads_lvm(
                        m, v, dm, dv, extra_res, cav_m, cav_v, alpha=alpha)
                    self.sgp_layer.update_factor(
                        idxs, alpha, grad_cav, extra_res, decay=decay)
                    self.update_factor_x(
                        idxs, alpha, grad_cav, cav_m, cav_v, decay=decay)

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'

    def compute_cavity_x(self, n, alpha):
        """Summary

        Args:
            n (TYPE): Description
            alpha (TYPE): Description

        Returns:
            TYPE: Description
        """
        # prior factor
        cav_x1 = self.t01 + (1 - alpha) * self.tx1[n, :]
        cav_x2 = self.t02 + (1 - alpha) * self.tx2[n, :]
        cav_v = 1.0 / cav_x2
        cav_m = cav_v * cav_x1
        return cav_m, cav_v, cav_x1, cav_x2

    def update_factor_x(self, n, alpha, grad_cav, cav_m, cav_v, decay=0.0):
        """Summary

        Args:
            n (TYPE): Description
            alpha (TYPE): Description
            grad_cav (TYPE): Description
            cav_m (TYPE): Description
            cav_v (TYPE): Description
            decay (float, optional): Description

        Returns:
            TYPE: Description
        """
        dmx = grad_cav['mx']
        dvx = grad_cav['vx']
        new_m = cav_m + cav_v * dmx
        new_v = cav_v - cav_v**2 * (dmx**2 - 2 * dvx)
        new_p2 = 1.0 / new_v
        new_p1 = new_p2 * new_m

        frac_t2 = new_p2 - 1.0 / cav_v
        frac_t1 = new_p1 - cav_m / cav_v
        # neg_idxs = np.where(frac_t2 < 0)
        # frac_t2[neg_idxs] = 0
        cur_t1 = self.tx1[n, :]
        cur_t2 = self.tx2[n, :]
        tx1_new = (1 - alpha) * cur_t1 + frac_t1
        tx2_new = (1 - alpha) * cur_t2 + frac_t2
        tx1_new = decay * cur_t1 + (1 - decay) * tx1_new
        tx2_new = decay * cur_t2 + (1 - decay) * tx2_new

        self.tx1[n, :] = tx1_new
        self.tx2[n, :] = tx2_new

    def get_posterior_x(self):
        """Summary

        Returns:
            TYPE: Description
        """
        post_1 = self.t01 + self.tx1
        post_2 = self.t02 + self.tx2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    # def predict_f(self, inputs):
    #     """Summary

    #     Args:
    #         inputs (TYPE): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     if not self.updated:
    #         self.sgp_layer.update_posterior()
    #         self.updated = True
    #     mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
    #     return mf, vf

    # def sample_f(self, inputs, no_samples=1):
    #     """Summary

    #     Args:
    #         inputs (TYPE): Description
    #         no_samples (int, optional): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     if not self.updated:
    #         self.sgp_layer.update_posterior()
    #         self.updated = True
    #     K = no_samples
    #     fs = np.zeros((inputs.shape[0], self.Dout, K))
    #     # TODO: remove for loop here
    #     for k in range(K):
    #         fs[:, :, k] = self.sgp_layer.sample(inputs)
    #     return fs

    # def predict_y(self, inputs):
    #     """Summary

    #     Args:
    #         inputs (TYPE): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     if not self.updated:
    #         self.sgp_layer.update_posterior()
    #         self.updated = True
    #     mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
    #     my, vy = self.lik_layer.output_probabilistic(mf, vf)
    #     return my, vy

    # def update_hypers(self, params):
    #     """Summary

    #     Args:
    #         params (TYPE): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     self.sgp_layer.update_hypers(params)
    #     self.lik_layer.update_hypers(params)

    # def init_hypers(self):
    #     """Summary

    #     Returns:
    #         TYPE: Description
    #     """
    #     sgp_params = self.sgp_layer.init_hypers()
    #     lik_params = self.lik_layer.init_hypers()
    #     init_params = dict(sgp_params)
    #     init_params.update(lik_params)

    #     return init_params

    # def get_hypers(self):
    #     """Summary

    #     Returns:
    #         TYPE: Description
    #     """
    #     sgp_params = self.sgp_layer.get_hypers()
    #     lik_params = self.lik_layer.get_hypers()
    #     params = dict(sgp_params)
    #     params.update(lik_params)
    #     return params


class SGPSSM(Base_SGPSSM):
    """Summary

    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        emi_layer (TYPE): Description
        lik (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        sgp_layer (TYPE): Description
        sn (int): Description
        updated (bool): Description
        x_next_1 (TYPE): Description
        x_next_2 (TYPE): Description
        x_prev_1 (TYPE): Description
        x_prev_2 (TYPE): Description
        x_prior_1 (TYPE): Description
        x_prior_2 (TYPE): Description
        x_up_1 (TYPE): Description
        x_up_2 (TYPE): Description
    """

    def __init__(self, y_train, hidden_size, no_pseudo,
                 lik='Gaussian', prior_mean=0, prior_var=1,
                 x_control=None, gp_emi=False, control_to_emi=True):
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
        super(SGPSSM, self).__init__(
            y_train, hidden_size, no_pseudo, 
            lik, prior_mean, prior_var,
            x_control, gp_emi, control_to_emi)

        self.dyn_layer = SGP_Layer(
            self.N - 1, self.Din + self.Dcon_dyn, self.Din, self.M)
        if gp_emi:
            self.emi_layer = SGP_Layer(
                self.N, self.Din + self.Dcon_emi, self.Dout, self.M)

        # natural params for latent variables
        N = self.N
        Din = self.Din
        self.x_prev_1 = np.zeros((N, Din))
        self.x_prev_2 = np.zeros((N, Din))
        self.x_next_1 = np.zeros((N, Din))
        self.x_next_2 = np.zeros((N, Din))
        self.x_up_1 = np.zeros((N, Din))
        self.x_up_2 = np.zeros((N, Din))
        self.x_prior_1 = prior_mean / prior_var
        self.x_prior_2 = 1.0 / prior_var
        self.UP, self.PREV, self.NEXT = 'UP', 'PREV', 'NEXT'

    def inf_parallel(self, epoch, alpha, decay):
        """Summary

        Args:
            epoch (TYPE): Description
            alpha (TYPE): Description
            decay (TYPE): Description

        Returns:
            TYPE: Description
        """
        # merge info from output
        cav_up_m, cav_up_v, _, _ = self.compute_cavity_x(self.UP, alpha)
        if not self.gp_emi:
            # only do this once at the begining for gaussian emission lik
            if isinstance(self.emi_layer, Gauss_Layer) and epoch == 0:
                up_1, up_2 = self.emi_layer.compute_factor(
                    cav_up_m, cav_up_v, alpha)
                self.x_up_1 = up_1
                self.x_up_2 = up_2
            else:
                up_1, up_2 = self.emi_layer.compute_factor(
                    cav_up_m, cav_up_v, alpha)
                self.x_up_1 = up_1
                self.x_up_2 = up_2
        # deal with the dynamics factors here
        cav_t_m, cav_t_v, cav_t_1, cav_t_2 = \
            self.compute_cavity_x(self.PREV, alpha)
        cav_tm1_m, cav_tm1_v, cav_tm1_1, cav_tm1_2 = \
            self.compute_cavity_x(self.NEXT, alpha)

        idxs = np.arange(self.N - 1)
        (mprop, vprop, extra_res) = \
            self.dyn_layer.forward_prop_thru_cav(
                idxs, cav_tm1_m, cav_tm1_v, alpha=alpha)
        logZ, dmprop, dvprop, dmt, dvt = \
            self.compute_transition_tilted(
                mprop, vprop, cav_t_m, cav_t_v, alpha)
        grad_hyper, grad_cav = self.dyn_layer.backprop_grads_lvm(
            mprop, vprop, dmprop, dvprop,
            extra_res, cav_tm1_m, cav_tm1_v, alpha=alpha)
        self.dyn_layer.update_factor(
            idxs, alpha, grad_cav, extra_res, decay=decay)

        self.update_factor_x(
            self.NEXT,
            grad_cav['mx'], grad_cav['vx'],
            cav_tm1_m, cav_tm1_v, cav_tm1_1, cav_tm1_2,
            decay=decay, alpha=alpha)
        self.update_factor_x(
            self.PREV,
            dmt, dvt,
            cav_t_m, cav_t_v, cav_t_1, cav_t_2,
            decay=decay, alpha=alpha)

    def inf_sequential(self, epoch, alpha, decay):
        """Summary

        Args:
            epoch (TYPE): Description
            alpha (TYPE): Description
            decay (TYPE): Description

        Returns:
            TYPE: Description
        """
        # merge info from output
        cav_up_m, cav_up_v, _, _ = self.compute_cavity_x(self.UP, alpha)
        if not self.gp_emi:
            # only do this once at the begining for gaussian emission lik
            if isinstance(self.emi_layer, Gauss_Layer) and epoch == 0:
                up_1, up_2 = self.emi_layer.compute_factor(
                    cav_up_m, cav_up_v, alpha)
                self.x_up_1 = up_1
                self.x_up_2 = up_2
            else:
                up_1, up_2 = self.emi_layer.compute_factor(
                    cav_up_m, cav_up_v, alpha)
                self.x_up_1 = up_1
                self.x_up_2 = up_2

        for n in range(0, self.N - 1):
            # deal with the dynamics factors here
            cav_t_m, cav_t_v, cav_t_1, cav_t_2 = \
                self.compute_cavity_x_sequential(self.PREV, [n + 1], alpha)
            cav_tm1_m, cav_tm1_v, cav_tm1_1, cav_tm1_2 = \
                self.compute_cavity_x_sequential(self.NEXT, [n], alpha)

            (mprop, vprop, extra_res) = \
                self.dyn_layer.forward_prop_thru_cav(
                    [n], cav_tm1_m, cav_tm1_v, alpha=alpha)
            logZ, dmprop, dvprop, dmt, dvt = \
                self.compute_transition_tilted(
                    mprop, vprop, cav_t_m, cav_t_v, alpha)
            grad_hyper, grad_cav = self.dyn_layer.backprop_grads_lvm(
                mprop, vprop, dmprop, dvprop,
                extra_res, cav_tm1_m, cav_tm1_v, alpha=alpha)
            self.dyn_layer.update_factor(
                [n], alpha, grad_cav, extra_res, decay=decay)

            self.update_factor_x_sequential(
                self.NEXT,
                grad_cav['mx'], grad_cav['vx'],
                cav_tm1_m, cav_tm1_v, cav_tm1_1, cav_tm1_2, [n],
                decay=decay, alpha=alpha)
            self.update_factor_x_sequential(
                self.PREV,
                dmt, dvt,
                cav_t_m, cav_t_v, cav_t_1, cav_t_2, [n + 1],
                decay=decay, alpha=alpha)

    def inference(self, alpha=1.0, no_epochs=10, parallel=True, decay=0):
        """Summary

        Args:
            alpha (float, optional): Description
            no_epochs (int, optional): Description
            parallel (bool, optional): Description
            decay (int, optional): Description

        Returns:
            TYPE: Description
        """
        try:
            for e in range(no_epochs):
                if e % 50 == 0:
                    print 'epoch %d/%d' % (e, no_epochs)
                if parallel:
                    self.inf_parallel(e, alpha, decay)
                else:
                    self.inf_sequential(e, alpha, decay)

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'

    def compute_cavity_x(self, mode, alpha):
        """Summary

        Args:
            mode (TYPE): Description
            alpha (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            NotImplementedError: Description
        """
        if mode == self.UP:
            cav_up_1 = self.x_prev_1 + self.x_next_1 + \
                (1 - alpha) * self.x_up_1
            cav_up_2 = self.x_prev_2 + self.x_next_2 + \
                (1 - alpha) * self.x_up_2
            cav_up_1[0, :] += self.x_prior_1
            cav_up_2[0, :] += self.x_prior_2
            return cav_up_1 / (cav_up_2 + 1e-16), 1.0 / (cav_up_2 + 1e-16), cav_up_1, cav_up_2
        elif mode == self.PREV:
            idxs = np.arange(1, self.N)
            cav_prev_1 = self.x_up_1[idxs, :] + self.x_next_1[idxs, :]
            cav_prev_2 = self.x_up_2[idxs, :] + self.x_next_2[idxs, :]
            cav_prev_1 += (1 - alpha) * self.x_prev_1[idxs, :]
            cav_prev_2 += (1 - alpha) * self.x_prev_2[idxs, :]
            return cav_prev_1 / cav_prev_2, 1.0 / cav_prev_2, cav_prev_1, cav_prev_2
        elif mode == self.NEXT:
            idxs = np.arange(0, self.N - 1)
            cav_next_1 = self.x_up_1[idxs, :] + self.x_prev_1[idxs, :]
            cav_next_2 = self.x_up_2[idxs, :] + self.x_prev_2[idxs, :]
            cav_next_1 += (1 - alpha) * self.x_next_1[idxs, :]
            cav_next_2 += (1 - alpha) * self.x_next_2[idxs, :]
            return cav_next_1 / cav_next_2, 1.0 / cav_next_2, cav_next_1, cav_next_2
        else:
            raise NotImplementedError('unknown mode')

    def compute_cavity_x_sequential(self, mode, idxs, alpha):
        """Summary

        Args:
            mode (TYPE): Description
            idxs (TYPE): Description
            alpha (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            NotImplementedError: Description
        """
        if mode == self.UP:
            cav_up_1 = self.x_prev_1 + self.x_next_1 + \
                (1 - alpha) * self.x_up_1
            cav_up_2 = self.x_prev_2 + self.x_next_2 + \
                (1 - alpha) * self.x_up_2
            cav_up_1[0, :] += self.x_prior_1
            cav_up_2[0, :] += self.x_prior_2
            return cav_up_1 / (cav_up_2 + 1e-16), 1.0 / (cav_up_2 + 1e-16), cav_up_1, cav_up_2
        elif mode == self.PREV:
            cav_prev_1 = self.x_up_1[idxs, :] + self.x_next_1[idxs, :]
            cav_prev_2 = self.x_up_2[idxs, :] + self.x_next_2[idxs, :]
            cav_prev_1 += (1 - alpha) * self.x_prev_1[idxs, :]
            cav_prev_2 += (1 - alpha) * self.x_prev_2[idxs, :]
            return cav_prev_1 / cav_prev_2, 1.0 / cav_prev_2, cav_prev_1, cav_prev_2
        elif mode == self.NEXT:
            cav_next_1 = self.x_up_1[idxs, :] + self.x_prev_1[idxs, :]
            cav_next_2 = self.x_up_2[idxs, :] + self.x_prev_2[idxs, :]
            cav_next_1 += (1 - alpha) * self.x_next_1[idxs, :]
            cav_next_2 += (1 - alpha) * self.x_next_2[idxs, :]
            return cav_next_1 / cav_next_2, 1.0 / cav_next_2, cav_next_1, cav_next_2
        else:
            raise NotImplementedError('unknown mode')

    def compute_transition_tilted(self, m_prop, v_prop, m_t, v_t, alpha):
        """Summary

        Args:
            m_prop (TYPE): Description
            v_prop (TYPE): Description
            m_t (TYPE): Description
            v_t (TYPE): Description
            alpha (TYPE): Description

        Returns:
            TYPE: Description
        """
        sn2 = np.exp(2 * self.sn)
        v_sum = v_t + v_prop + sn2 / alpha
        m_diff = m_t - m_prop
        exp_term = -0.5 * m_diff**2 / v_sum
        const_term = -0.5 * np.log(2 * np.pi * v_sum)
        alpha_term = 0.5 * (1 - alpha) * np.log(2 *
                                                np.pi * sn2) - 0.5 * np.log(alpha)
        logZ = exp_term + const_term + alpha_term

        dvt = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dvprop = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dmt = m_diff / v_sum
        dmprop = m_diff / v_sum
        return logZ, dmprop, dvprop, dmt, dvt

    def update_factor_x(
            self, mode, dmcav, dvcav, mcav, vcav, n1cav, n2cav,
            decay=0.0, alpha=1.0):
        """Summary

        Args:
            mode (TYPE): Description
            dmcav (TYPE): Description
            dvcav (TYPE): Description
            mcav (TYPE): Description
            vcav (TYPE): Description
            n1cav (TYPE): Description
            n2cav (TYPE): Description
            decay (float, optional): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description

        Raises:
            NotImplementedError: Description
        """
        new_m = mcav + vcav * dmcav
        new_v = vcav - vcav**2 * (dmcav**2 - 2 * dvcav)
        new_n2 = 1.0 / new_v
        new_n1 = new_n2 * new_m
        frac_n2 = new_n2 - n2cav
        frac_n1 = new_n1 - n1cav
        if mode == self.NEXT:
            idxs = np.arange(0, self.N - 1)
            cur_n1 = self.x_next_1[idxs, :]
            cur_n2 = self.x_next_2[idxs, :]
            n1_new = (1 - alpha) * cur_n1 + frac_n1
            n2_new = (1 - alpha) * cur_n2 + frac_n2
            self.x_next_1[idxs, :] = decay * cur_n1 + (1 - decay) * n1_new
            self.x_next_2[idxs, :] = decay * cur_n2 + (1 - decay) * n2_new
        elif mode == self.PREV:
            idxs = np.arange(1, self.N)
            cur_n1 = self.x_prev_1[idxs, :]
            cur_n2 = self.x_prev_2[idxs, :]
            n1_new = (1 - alpha) * cur_n1 + frac_n1
            n2_new = (1 - alpha) * cur_n2 + frac_n2
            self.x_prev_1[idxs, :] = decay * cur_n1 + (1 - decay) * n1_new
            self.x_prev_2[idxs, :] = decay * cur_n2 + (1 - decay) * n2_new
        else:
            raise NotImplementedError('unknown mode')

    def update_factor_x_sequential(
            self, mode, dmcav, dvcav, mcav, vcav, n1cav, n2cav,
            idxs, decay=0.0, alpha=1.0):
        """Summary

        Args:
            mode (TYPE): Description
            dmcav (TYPE): Description
            dvcav (TYPE): Description
            mcav (TYPE): Description
            vcav (TYPE): Description
            n1cav (TYPE): Description
            n2cav (TYPE): Description
            idxs (TYPE): Description
            decay (float, optional): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description

        Raises:
            NotImplementedError: Description
        """
        new_m = mcav + vcav * dmcav
        new_v = vcav - vcav**2 * (dmcav**2 - 2 * dvcav)
        new_n2 = 1.0 / new_v
        new_n1 = new_n2 * new_m
        frac_n2 = new_n2 - n2cav
        frac_n1 = new_n1 - n1cav
        if mode == self.NEXT:
            cur_n1 = self.x_next_1[idxs, :]
            cur_n2 = self.x_next_2[idxs, :]
            n1_new = (1 - alpha) * cur_n1 + frac_n1
            n2_new = (1 - alpha) * cur_n2 + frac_n2
            self.x_next_1[idxs, :] = decay * cur_n1 + (1 - decay) * n1_new
            self.x_next_2[idxs, :] = decay * cur_n2 + (1 - decay) * n2_new
        elif mode == self.PREV:
            cur_n1 = self.x_prev_1[idxs, :]
            cur_n2 = self.x_prev_2[idxs, :]
            n1_new = (1 - alpha) * cur_n1 + frac_n1
            n2_new = (1 - alpha) * cur_n2 + frac_n2
            self.x_prev_1[idxs, :] = decay * cur_n1 + (1 - decay) * n1_new
            self.x_prev_2[idxs, :] = decay * cur_n2 + (1 - decay) * n2_new
        else:
            raise NotImplementedError('unknown mode')

    def get_posterior_x(self):
        """Summary

        Returns:
            TYPE: Description
        """
        post_1 = self.x_next_1 + self.x_prev_1 + self.x_up_1
        post_2 = self.x_next_2 + self.x_prev_2 + self.x_up_2
        post_1[0, :] += self.x_prior_1
        post_2[0, :] += self.x_prior_2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def get_posterior_y(self):
        """Summary

        Returns:
            TYPE: Description
        """
        mx, vx = self.get_posterior_x()
        my, vy, vyn = self.emi_layer.output_probabilistic(mx, vx)
        return my, vy, vyn

    # def predict_f(self, inputs):
    #     """Summary

    #     Args:
    #         inputs (TYPE): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     if not self.updated:
    #         self.sgp_layer.update_posterior()
    #         self.updated = True
    #     mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
    #     return mf, vf

    # def sample_f(self, inputs, no_samples=1):
    #     """Summary

    #     Args:
    #         inputs (TYPE): Description
    #         no_samples (int, optional): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     if not self.updated:
    #         self.sgp_layer.update_posterior()
    #         self.updated = True
    #     K = no_samples
    #     fs = np.zeros((inputs.shape[0], self.Dout, K))
    #     # TODO: remove for loop here
    #     for k in range(K):
    #         fs[:, :, k] = self.sgp_layer.sample(inputs)
    #     return fs

    # def predict_y(self, inputs):
    #     """Summary

    #     Args:
    #         inputs (TYPE): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     if not self.updated:
    #         self.sgp_layer.update_posterior()
    #         self.updated = True
    #     mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
    #     my, vy = self.emi_layer.output_probabilistic(mf, vf)
    #     return my, vy

    # def update_hypers(self, params):
    #     """Summary

    #     Args:
    #         params (TYPE): Description

    #     Returns:
    #         TYPE: Description
    #     """
    #     self.sgp_layer.update_hypers(params)
    #     self.emi_layer.update_hypers(params)
    #     self.sn = params['sn']

    # def init_hypers(self):
    #     """Summary

    #     Returns:
    #         TYPE: Description
    #     """
    #     sgp_params = self.sgp_layer.init_hypers()
    #     lik_params = self.emi_layer.init_hypers()
    #     ssm_params = {'sn': np.log(0.001)}
    #     init_params = dict(sgp_params)
    #     init_params.update(lik_params)
    #     init_params.update(ssm_params)
    #     return init_params

    # def get_hypers(self):
    #     """Summary

    #     Returns:
    #         TYPE: Description
    #     """
    #     sgp_params = self.sgp_layer.get_hypers()
    #     emi_params = self.emi_layer.get_hypers()
    #     ssm_params = {'sn': self.sn}
    #     params = dict(sgp_params)
    #     params.update(emi_params)
    #     params.update(ssm_params)
    #     return params


class SGP_Layer_rank_one(object):
    """Summary

    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        Kuu (TYPE): Description
        Kuuinv (TYPE): Description
        ls (TYPE): Description
        M (TYPE): Description
        mu (TYPE): Description
        N (TYPE): Description
        sf (int): Description
        Splusmm (TYPE): Description
        Su (TYPE): Description
        Suinv (TYPE): Description
        SuinvMu (TYPE): Description
        t1 (TYPE): Description
        t2 (TYPE): Description
        zu (TYPE): Description
    """

    def __init__(self, no_train, input_size, output_size, no_pseudo):
        """Summary

        Args:
            no_train (TYPE): Description
            input_size (TYPE): Description
            output_size (TYPE): Description
            no_pseudo (TYPE): Description
        """
        self.Din = Din = input_size
        self.Dout = Dout = output_size
        self.M = M = no_pseudo
        self.N = N = no_train

        # factor variables
        self.variances = np.zeros([N, Dout])
        self.variances.fill(1e20)
        self.means = np.zeros([N, Dout])

        # pep variables
        self.gamma = np.zeros([Dout, M])
        self.beta = np.zeros([Dout, M, M])
        
        # numpy variable for inducing points, Kuuinv, Kuu and its gradients
        self.zu = np.zeros([M, Din])
        self.Kuu = np.zeros([M, M])
        self.Kuuinv = np.zeros([M, M])

        # variables for the hyperparameters
        self.ls = np.zeros([Din, ])
        self.sf = 0

    def forward_prop_thru_post(self, x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        Kuuinv = self.Kuuinv
        A = np.einsum('ab,db->da', Kuuinv, self.mu)
        B = np.einsum(
            'ab,dbc->dac',
            Kuuinv, np.einsum('dab,bc->dac', self.Su, Kuuinv)) - Kuuinv
        kff = np.exp(2 * self.sf)
        kfu = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', kfu, A)
        Bpsi2 = np.einsum('dab,na,nb->nd', B, kfu, kfu)
        vout = kff + Bpsi2
        return mout, vout

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
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def update_posterior(self, x_train=None, new_hypers=False):
        """Summary

        Returns:
            TYPE: Description
        """
        # compute the posterior approximation
        if new_hypers and x_train is not None:
            Kfu = compute_kernel(2*self.ls, 2*self.sf, x_train, self.zu)
            KuuinvKuf = np.dot(self.Kuuinv, Kfu.T)
            self.Kfu = Kfu
            self.KuuinvKuf = KuuinvKuf
            self.Kff_diag = compute_kernel_diag(2*self.ls, 2*self.sf, x_train)

        KuuinvKuf_div_var = np.einsum('an,nd->dan', self.KuuinvKuf, 1.0 / self.variances)
        T2u = np.einsum('dan,bn->dab', KuuinvKuf_div_var, self.KuuinvKuf)
        T1u = np.einsum('bn,nd->db', self.KuuinvKuf, self.means / self.variances)
        Vinv = self.Kuuinv + T2u
        self.Suinv = Vinv
        self.Su = np.linalg.inv(Vinv)
        self.mu = np.einsum('dab,db->da', self.Su, T1u)
        self.gamma = np.einsum('ab,db->da', self.Kuuinv, self.mu)
        self.beta = self.Kuuinv - np.einsum('ab,dbc->dac', 
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
                ls[i] = np.log(d2imed + 1e-16)
            sf = np.log(np.array([0.5]))

        params = dict()
        params['sf' + key_suffix] = sf
        params['ls' + key_suffix] = ls
        params['zu' + key_suffix] = zu

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
        params_zu_i = self.zu
        params['zu' + key_suffix] = self.zu
        return params

    def update_hypers(self, params, x_train, key_suffix=''):
        """Summary

        Args:
            params (TYPE): Description
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
        self.ls = params['ls' + key_suffix]
        self.sf = params['sf' + key_suffix]
        self.zu = params['zu' + key_suffix]

        # update Kuu given new hypers
        self.compute_kuu()
        # compute mu and Su for each layer
        self.update_posterior(x_train, new_hypers=True)

    def compute_cavity(self, idxs, alpha):
        # deletion
        p_i = self.KuuinvKuf[:, idxs].T[:, np.newaxis, :]
        k_i = self.Kfu[idxs, :]
        k_ii = self.Kff_diag[idxs][:, np.newaxis]
        gamma = self.gamma
        beta = self.beta
        h_si = p_i - np.einsum('dab,nb->nda', beta, k_i)
        variance_i = self.variances[idxs, :]
        mean_i = self.means[idxs, :]
        dlogZd_dmi2 = 1.0 / (variance_i/alpha - 
            np.sum(k_i[:, np.newaxis, :] * h_si, axis=2))
        dlogZd_dmi = -dlogZd_dmi2 * (mean_i - 
            np.sum(k_i[:, np.newaxis, :] * gamma, axis=2))
        hd1 = h_si * dlogZd_dmi[:, :, np.newaxis]
        hd2h = np.einsum('nda,ndb->ndab', h_si, h_si) * dlogZd_dmi2[:, :, np.newaxis, np.newaxis]
        gamma_si = gamma + hd1
        beta_si = beta - hd2h

        # projection
        h = p_i - np.einsum('ndab,nb->nda', beta_si, k_i)
        m_si_i = np.einsum('na,nda->nd', k_i, gamma_si)
        v_si_ii = k_ii - np.einsum('na,ndab,nb->nd', k_i, beta_si, k_i)

        return m_si_i, v_si_ii, [h, beta_si, gamma_si]

    def update_factor(self, idxs, mcav, vcav, dm, dm2, dv, alpha, prop_info):
        h, beta_si, gamma_si = prop_info[0], prop_info[1], prop_info[2]
        k_i = self.Kfu[idxs, :]
        variance_i = self.variances[idxs, :]
        mean_i = self.means[idxs, :]
        
        var_i_new = -1.0 / dm2 - np.sum(k_i[:, np.newaxis, :] * h, axis=2)
        mean_i_new = mcav - dm / dm2

        var_new_parallel = 1 / (1 / var_i_new + 1 / variance_i * (1 - alpha))
        mean_div_var_i_new = (mean_i_new / var_i_new + 
            mean_i / variance_i * (1 - alpha))
        mean_new_parallel = mean_div_var_i_new * var_new_parallel

        # if alpha == 1:
        #     rho = 0.5
        #     n1_new = 1.0 / var_new_parallel
        #     n2_new = mean_new_parallel / var_new_parallel
        #     n1_ori = 1.0 / variance_i
        #     n2_ori = mean_i / variance_i
        #     n1_damped = rho * n1_new + (1.0 - rho) * n1_ori
        #     n2_damped = rho * n2_new + (1.0 - rho) * n2_ori
        #     var_new_parallel = 1.0 / n1_damped
        #     mean_new_parallel = var_new_parallel * n2_damped

        self.variances[idxs, :] = var_new_parallel
        self.means[idxs, :] = mean_new_parallel

    def backprop_grads_reg(self, idxs, m, v, dm, dm2, dv, x, alpha, prop_info):
        N = self.N
        Nb = idxs.shape[0]
        sf2 = np.exp(2 * self.sf)
        Dout = self.Dout
        Kuu = self.Kuu
        Kuuinv = self.Kuuinv
        Su = self.Su
        mu = self.mu
        Suinv = self.Suinv
        p_i = self.KuuinvKuf[:, idxs].T
        h, beta_si, gamma_si = prop_info[0], prop_info[1], prop_info[2]
        kfu = self.Kfu[idxs, :]
        variance_i = self.variances[idxs, :]
        mean_i = self.means[idxs, :]
       
        # compute cavity covariance
        betacavKuu = np.einsum('ndab,bc->ndac', beta_si, Kuu)
        mcav = np.einsum('ab,ndb->nda', Kuu, gamma_si)
        Sucav = Kuu - np.einsum('ab,ndbc->ndac', Kuu, betacavKuu)
        signSu, logdetSu = np.linalg.slogdet(Su)
        signKuu, logdetKuu = np.linalg.slogdet(Kuu)
        Suinvm = np.einsum('dab,db->da', Suinv, mu)
        term1 = 0.5 * (np.sum(logdetSu) - Dout * logdetKuu + np.sum(mu * Suinvm))

        variance_i = self.variances[idxs, :]
        mean_i = self.means[idxs, :]
        tn = 1.0 / variance_i
        gn = mean_i
        wnScav = np.einsum('na,ndab->ndb', p_i, Sucav)
        wnScavwn = np.einsum('ndb,nb->nd', wnScav, p_i)
        wnScavSinvm = np.sum(wnScav * Suinvm, axis=2)
        wnS = np.einsum('na,dab->ndb', p_i, Su)
        wnSwn = np.sum(wnS * p_i[:, np.newaxis, :], axis=2)
        mwn = np.sum(mu * p_i[:, np.newaxis, :], axis=2)
        oneminuswnSwn = 1 - alpha * tn * wnSwn

        term2a = 0.5 * alpha * tn**2 * gn**2 * wnScavwn
        term2b = - gn * tn * wnScavSinvm
        term2c = 0.5 * tn * mwn**2 / oneminuswnSwn
        term2d = -0.5 / alpha * np.log(oneminuswnSwn)
        term2 = N / Nb * np.sum(term2a + term2b + term2c + term2d)

        sgp_contrib = - term1 - term2

        KuuinvMcav = np.einsum('ab,ndb->nda', Kuuinv, mcav)
        dmiKuuinvMcav = dm[:, :, np.newaxis] * KuuinvMcav
        dKuu_via_mi = -np.einsum('nda,nb->ab', dmiKuuinvMcav, p_i)
        
        VcavKuuinvKufi = np.einsum('ndab,nb->nda', Sucav, p_i)
        KuuinvVcavKuuinvKufi = np.einsum('ab,ndb->nda', Kuuinv, VcavKuuinvKufi)
        p_idlogZ_dvi = p_i[:, np.newaxis, :] * dv[:, :, np.newaxis]
        temp1 = - np.einsum('nda,ndb->ab', KuuinvVcavKuuinvKufi, p_idlogZ_dvi)
        temp2 = np.transpose(temp1, [0, 1])
        temp3 = np.einsum('na,ndb->ab', p_i, p_idlogZ_dvi)
        dKuu_via_vi = temp1 + temp2 + temp3
        dKuu_via_logZ = dKuu_via_mi + dKuu_via_vi

        dKfu_via_mi = dmiKuuinvMcav
        dKfu_via_vi = 2 * dv[:, :, np.newaxis] * (-p_i[:, np.newaxis, :] + KuuinvVcavKuuinvKufi)
        dKfu_via_logZ = np.sum(dKfu_via_mi + dKfu_via_vi, axis=1)
        dsf2, dls, dzu = compute_kfu_derivatives(
            dKfu_via_logZ, kfu, np.exp(self.ls), sf2, x, self.zu)
        dls = dls * np.exp(self.ls)
        dsf2 += np.sum(dv)
        dsf = 2 * sf2 * dsf2

        # compute the gradients
        Vmm = Su + np.einsum('da,db->dab', mu, mu)
        S = self.Dout * Kuuinv - np.sum(np.einsum('ab,dbc->dac', 
            Kuuinv, np.einsum('dab,bc->dac', Vmm, Kuuinv)), axis=0)
        S = 0.5 * S + dKuu_via_logZ
        dhyp = d_trace_MKzz_dhypers(
            2*self.ls, 2*self.sf, self.zu, S, 
            Kuu - np.diag(JITTER * np.ones(self.M)))

        grads = {}
        grads['sf'] = 2*dhyp[0] + dsf
        grads['ls'] = 2*dhyp[1] + dls
        grads['zu'] = dhyp[2] + dzu

        return sgp_contrib, grads


class SGPR_rank_one(Base_SGPR):
    """Summary

    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        lik_layer (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        sgp_layer (TYPE): Description
        updated (bool): Description
        x_train (TYPE): Description
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
        super(SGPR_rank_one, self).__init__(x_train, y_train, no_pseudo, lik)
        self.sgp_layer = SGP_Layer_rank_one(self.N, self.Din, self.Dout, self.M)

    def objective_function(self, params, mb_size, alpha=1.0, prop_mode=PROP_MM):
        """Summary
        
        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            alpha (float, optional): Description
            prop_mode (TYPE, optional): Description
        
        Returns:
            TYPE: Description
        """
        N = self.N
        if mb_size >= N:
            idxs = np.arange(N)
            xb = self.x_train
            yb = self.y_train
        else:
            idxs = np.random.choice(N, mb_size, replace=False)
            xb = self.x_train[idxs, :]
            yb = self.y_train[idxs, :]
        # update model with new hypers
        self.update_hypers(params)
        # run power-EP and compute grads
        no_ep_sweeps = 10 # TODO: put this in config
        parallel = True # TODO: put this in config
        energy, grad_all = self.run_pep(idxs, no_ep_sweeps, alpha, parallel, 
            compute_energy=True)
        return energy, grad_all

    def run_pep(self, train_idxs, no_sweeps, alpha, parallel, compute_energy=False, display_steps=10):
        if parallel:
            return self.run_pep_parallel(
                train_idxs, no_sweeps, alpha, compute_energy, display_steps)
        else:
            # TODO
            return self.run_pep_sequential(
                train_idxs, no_sweeps, alpha, compute_energy, display_steps)

    def run_pep_parallel(self, idxs, no_sweeps, alpha, compute_energy, display_steps):
        batch_size = idxs.shape[0]
        scale_logZ = - self.N * 1.0 / batch_size / alpha
        # scale_logZ = 0
        xb = self.x_train[idxs, :]
        yb = self.y_train[idxs, :]
        energy = {}
        grad_all = {}
        for k in range(no_sweeps):
            # if k % display_steps == 0:
            #     print 'PEP, epoch: %d / %d' % (k, no_sweeps)
            find_log_lik = compute_energy and (k == no_sweeps-1)
            m_cav, v_cav, prop_info = self.sgp_layer.compute_cavity(idxs, alpha)
            logZ, dm, dv, dm2 = self.lik_layer.compute_log_Z(
                m_cav, v_cav, yb, alpha, compute_dm2=True)
            # dm *= 0
            # dm2 *= 0
            # dm2 += 1e-16
            # dv *= 0
            self.sgp_layer.update_factor(
                idxs, m_cav, v_cav, dm, dm2, dv, alpha, prop_info)
            self.sgp_layer.update_posterior(None, new_hypers=False)

            if find_log_lik:
                N = self.N
                lik_contrib = scale_logZ * np.sum(logZ)
                dm_s = scale_logZ * dm
                dv_s = scale_logZ * dv
                dm2_s = scale_logZ * dm2
                sgp_contrib, sgp_grad = self.sgp_layer.backprop_grads_reg(
                    idxs, m_cav, v_cav, dm_s, dm2_s, dv_s, xb, alpha, prop_info)
                lik_grad = self.lik_layer.backprop_grads(
                    m_cav, v_cav, dm, dv, alpha, scale_logZ)
                energy = sgp_contrib + lik_contrib

                grad_all = {}
                for key in sgp_grad.keys():
                    grad_all[key] = sgp_grad[key]

                for key in lik_grad.keys():
                    grad_all[key] = lik_grad[key]

                energy /= N
                for key in grad_all.keys():
                    grad_all[key] /= N

        return energy, grad_all

    def run_pep_sequential(self, idxs, no_sweeps, alpha, compute_energy, 
        display_steps):
        batch_size = idxs.shape[0]
        scale_logZ = - self.N * 1.0 / batch_size / alpha
        xb = self.x_train[idxs, :]
        yb = self.y_train[idxs, :]
        for k in range(no_sweeps):
            if k % display_steps == 0:
                print 'PEP, epoch: %d / %d' % (k, no_sweeps)
            find_log_lik = compute_energy and (k == no_sweeps-1)
            for i in range(batch_size):
                m_cav, v_cav, prop_info = self.sgp_layer.compute_cavity(
                    [idxs[i]], alpha)
                logZ, dm, dv, dm2 = self.lik_layer.compute_log_Z(
                    m_cav, v_cav, yb[i], alpha, compute_dm2=True)
                self.sgp_layer.update_factor(
                    [idxs[i]], m_cav, v_cav, dm, dm2, dv, alpha, prop_info)
                self.sgp_layer.update_posterior(None, new_hypers=False)

    def update_hypers(self, params):
        """Summary
        
        Args:
            params (TYPE): Description
        """
        self.sgp_layer.update_hypers(params, self.x_train)
        self.lik_layer.update_hypers(params)

