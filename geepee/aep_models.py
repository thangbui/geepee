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
import pprint
from utils import profile
import cPickle as pickle

pp = pprint.PrettyPrinter(indent=4)

from utils import *
from kernels import *

# TODO: ideally these should be moved to some config file
jitter = 1e-5
gh_degree = 10
# TODO: do replacement sampling for now since this is faster
# alternative would be
# idxs = np.random.permutation(N)[:mb_size]


class SGP_Layer(object):

    def __init__(self, no_train, input_size, output_size, no_pseudo):
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

        # variables for the cavity distribution
        self.muhat = np.zeros([Dout, M, ])
        self.Suhat = np.zeros([Dout, M, M])
        self.Suhatinv = np.zeros([Dout, M, M])
        self.Splusmmhat = np.zeros([Dout, M, M])

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
        self.Ahat = np.zeros([Dout, M, ])
        self.Bhat_sto = np.zeros([Dout, M, M])
        self.Bhat_det = np.zeros([Dout, M, M])
        self.A = np.zeros([Dout, M, ])
        self.B_det = np.zeros([Dout, M, M])
        self.B_sto = np.zeros([Dout, M, M])

    def compute_phi_prior(self):
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        logZ_prior = self.Dout * 0.5 * logdet
        return logZ_prior

    def compute_phi_posterior(self):
        (sign, logdet) = np.linalg.slogdet(self.Su)
        phi_posterior = 0.5 * np.sum(logdet)
        phi_posterior += 0.5 * np.sum(self.mu*np.linalg.solve(
            self.Su, self.mu))
        return phi_posterior

    def compute_phi_cavity(self):
        logZ_posterior = 0
        (sign, logdet) = np.linalg.slogdet(self.Suhat)
        phi_cavity = 0.5 * np.sum(logdet)
        phi_cavity += 0.5 * np.sum(self.muhat*np.linalg.solve(
            self.Suhat, self.muhat))
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

    def forward_prop_thru_cav(self, mx, vx=None):
        if vx is None:
            return self._forward_prop_deterministic_thru_cav(mx)
        else:
            return self._forward_prop_random_thru_cav_mm(mx, vx)

    def _forward_prop_deterministic_thru_cav(self, x):
        kff = np.exp(2*self.sf)
        kfu = compute_kernel(2*self.ls, 2*self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', kfu, self.Ahat)
        Bkfukuf = np.einsum('dab,na,nb->nd', self.Bhat_det, kfu, kfu)
        vout = kff + Bkfukuf
        return mout, vout, kfu

    @profile
    def _forward_prop_random_thru_cav_mm(self, mx, vx):
        psi0 = np.exp(2*self.sf)
        psi1, psi2 = compute_psi_weave(2*self.ls, 2*self.sf, mx, vx, self.zu)
        # psi1 = compute_psi1(2*self.ls, 2*self.sf, mx, vx, self.zu)
        # psi2 = compute_psi2(2*self.ls, 2*self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.Ahat)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.Bhat_sto, psi2)
        vout = psi0 + Bhatpsi2 - mout**2
        return mout, vout, psi1, psi2

    def forward_prop_thru_post(self, mx, vx=None):
        if vx is None:
            return self._forward_prop_deterministic_thru_post(mx)
        else:
            return self._forward_prop_random_thru_post_mm(mx, vx)

    def _forward_prop_deterministic_thru_post(self, x):
        psi0 = np.exp(2*self.sf)
        psi1 = compute_kernel(2*self.ls, 2*self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bpsi2 = np.einsum('dab,na,nb->nd', self.B_det, psi1, psi1)
        vout = psi0 + Bpsi2
        return mout, vout

    def _forward_prop_random_thru_post_mm(self, mx, vx):
        psi0 = np.exp(2.0*self.sf)
        psi1, psi2 = compute_psi_weave(2*self.ls, 2*self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bpsi2 = np.einsum('dab,nab->nd', self.B_sto, psi2)
        vout = psi0 + Bpsi2 - mout**2
        return mout, vout

    @profile
    def backprop_grads_lvm(self, m, v, dm, dv, psi1, psi2, mx, vx, alpha=1.0):
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2*self.sf)
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        mu = self.mu
        Su = self.Su
        Spmm = self.Splusmm
        muhat = self.muhat
        Suhat = self.Suhat
        Spmmhat = self.Splusmmhat
        Kuuinv = self.Kuuinv

        beta = (N - alpha) * 1.0 / N
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        # compute grads wrt Ahat and Bhat
        dm_all = dm - 2 * dv * m
        dAhat = np.einsum('nd,nm->dm', dm_all, psi1)
        dBhat = np.einsum('nd,nab->dab', dv, psi2)
        # compute grads wrt psi1 and psi2
        dpsi1 = np.einsum('nd,dm->nm', dm_all, self.Ahat)
        dpsi2 = np.einsum('nd,dab->nab', dv, self.Bhat_sto)
        dsf2, dls, dzu, dmx, dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, ls, sf2, mx, vx, self.zu)

        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        dvcav = np.einsum('ab,dbc,ce->dae', Kuuinv, dBhat, Kuuinv)
        dmcav = 2 * np.einsum('dab,db->da', dvcav, muhat) \
            + np.einsum('ab,db->da', Kuuinv, dAhat)

        dvcav_via_mcav = beta * np.einsum('da,db->dab', dmcav, self.theta_2)
        dvcav += dvcav_via_mcav
        dvcavinv = - np.einsum('dab,dbc,dce->dae', Suhat, dvcav, Suhat)
        dtheta1 = beta * dvcavinv
        dtheta2 = beta * np.einsum('dab,db->da', Suhat, dmcav)
        dKuuinv_via_vcav = np.sum(dvcavinv, axis=0)

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dKuuinv_via_Ahat = np.einsum('da,db->ab', dAhat, muhat)
        KuuinvSmmd = np.einsum('ab,dbc->dac', Kuuinv, Spmmhat)
        dKuuinv_via_Bhat = 2 * np.einsum('dab,dac->bc', KuuinvSmmd, dBhat) \
            - np.sum(dBhat, axis=0)
        dKuuinv = dKuuinv_via_Ahat + dKuuinv_via_Bhat + dKuuinv_via_vcav
        Minner = scale_poste * np.sum(Spmm, axis=0) + scale_cav * \
            np.sum(Spmmhat, axis=0) - 2.0 * dKuuinv

        dtheta1 = -0.5*scale_poste*Spmm - 0.5*scale_cav*beta*Spmmhat + dtheta1
        dtheta2 = scale_poste*mu + scale_cav*beta*muhat + dtheta2
        dtheta1T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum('dab,dbc->dac', self.theta_1_R, dtheta1+dtheta1T)

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

        M_all = 0.5 * (scale_prior*self.Dout*Kuuinv +
                       np.dot(Kuuinv, np.dot(Minner, Kuuinv)))
        dhyp = d_trace_MKzz_dhypers(
            2*self.ls, 2*self.sf, self.zu, M_all, self.Kuu)

        dzu += dhyp[2]
        dls += 2*dhyp[1]
        dsf += 2*dhyp[0]

        grad_hyper = {
            'sf': dsf, 'ls': dls, 'zu': dzu, 
            'eta1_R': deta1_R, 'eta2': deta2} 
        grad_input = {'mx': dmx, 'vx': dvx}

        return grad_hyper, grad_input

    @profile
    def backprop_grads_reg(self, m, v, dm, dv, kfu, x, alpha=1.0):
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2*self.sf)
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

        beta = (N - alpha) * 1.0 / N
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1

        # compute grads wrt kfu
        dkfu_m = np.einsum('nd,dm->nm', dm, self.Ahat)
        dkfu_v = 2*np.einsum('nd,dab,na->nb', dv, self.Bhat_det, kfu)
        dkfu = dkfu_m + dkfu_v
        dsf2, dls, dzu = compute_kfu_derivatives(
            dkfu, kfu, ls, sf2, x, self.zu)
        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        # compute grads wrt theta1 and theta2
        SKuuinvKuf = np.einsum('dab,nb->nda', Suhat, kfuKuuinv)
        dSinv_via_v = - np.einsum('nda,nd,ndb->dab', SKuuinvKuf, dv, SKuuinvKuf)
        dSinv_via_m = - np.einsum('nda,nd,db->dab', SKuuinvKuf, dm, muhat)
        dSinv = dSinv_via_m + dSinv_via_v
        dSinvM = np.einsum('nda,nd->da', SKuuinvKuf, dm)
        dtheta1 = beta * dSinv
        dtheta2 = beta * dSinvM

        dtheta1 = -0.5*scale_poste*Spmm - 0.5*scale_cav*beta*Spmmhat + dtheta1
        dtheta2 = scale_poste*mu + scale_cav*beta*muhat + dtheta2
        dtheta1T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum('dab,dbc->dac', self.theta_1_R, dtheta1+dtheta1T)

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
        dKuuinv_v_1 = 2*np.einsum('dab,dac->bc', KuuinvSmmd, dBhat) \
            - np.sum(dBhat, axis=0)
        dKuuinv_v_2 = np.sum(dSinv, axis=0)
        dKuuinv = dKuuinv_m + dKuuinv_v_1 + dKuuinv_v_2
        
        Minner = scale_poste * np.sum(Spmm, axis=0) + scale_cav * \
            np.sum(Spmmhat, axis=0) - 2.0 * dKuuinv
        M_all = 0.5 * (scale_prior*self.Dout*Kuuinv +
                       np.dot(Kuuinv, np.dot(Minner, Kuuinv)))
        dhyp = d_trace_MKzz_dhypers(
            2*self.ls, 2*self.sf, self.zu, M_all, self.Kuu)

        dzu += dhyp[2]
        dls += 2*dhyp[1]
        dsf += 2*dhyp[0]

        grad_hyper = {
            'sf': dsf, 'ls': dls, 'zu': dzu, 
            'eta1_R': deta1_R, 'eta2': deta2 
            }

        return grad_hyper

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
        # self.Kuuinv = matrixInverse(self.Kuu)
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def compute_cavity(self, alpha=1.0):
        # compute the leave one out moments
        beta = (self.N - alpha) * 1.0 / self.N
        Dout = self.Dout
        Kuu = self.Kuu
        Kuuinv = self.Kuuinv
        self.Suhatinv = Kuuinv + beta*self.theta_1
        self.Suhat = np.linalg.inv(self.Suhatinv)
        self.muhat = np.einsum('dab,db->da', self.Suhat, beta*self.theta_2)
        self.Ahat = np.einsum('ab,db->da', Kuuinv, self.muhat)
        Smm = self.Suhat + np.einsum('da,db->dab', self.muhat, self.muhat)
        self.Splusmmhat = Smm
        self.Bhat_sto = - Kuuinv + np.einsum(
            'ab,dbc->dac',
            Kuuinv, 
            np.einsum('dab,bc->dac', Smm, Kuuinv))
        self.Bhat_det = - Kuuinv + np.einsum(
            'ab,dbc->dac',
            Kuuinv, 
            np.einsum('dab,bc->dac', self.Suhat, Kuuinv))

    def update_posterior(self):
        # compute the posterior approximation
        for d in range(self.Dout):
            Sinv = self.Kuuinv + self.theta_1[d, :, :]
            SinvM = self.theta_2[d, :]
            # S = matrixInverse(Sinv)
            S = np.linalg.inv(Sinv)
            self.Su[d, :, :] = S
            m = np.dot(S, SinvM)
            self.mu[d, :] = m

            Smm = S + np.outer(m, m)
            self.Splusmm[d, :, :] = Smm

    def update_posterior_for_prediction(self):
        # compute the posterior approximation
        Kuuinv = self.Kuuinv
        for d in range(self.Dout):
            Sinv = Kuuinv + self.theta_1[d, :, :]
            SinvM = self.theta_2[d, :]
            # S = matrixInverse(Sinv)
            S = np.linalg.inv(Sinv)
            self.Su[d, :, :] = S
            m = np.dot(S, SinvM)
            self.mu[d, :] = m

            self.A[d, :] = np.dot(Kuuinv, m)
            Smm = S + np.outer(m, m)
            self.Splusmm[d, :, :] = Smm
            self.B_det[d, :, :] = np.dot(Kuuinv, np.dot(S, Kuuinv)) - Kuuinv
            self.B_sto[d, :, :] = np.dot(Kuuinv, np.dot(Smm, Kuuinv)) - Kuuinv

    def init_hypers(self, x_train=None, key_suffix=''):
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
                ls[i] = np.log(d2imed/2  + 1e-16)
            sf = np.log(np.array([0.5]))

        Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        Kuu += np.diag(jitter * np.ones((M, )))
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
        # compute muhat and Suhat for each layer
        # self.compute_cavity(alpha=alpha)


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

    def output_probabilistic(self, mf, vf):
        return mf, vf + np.exp(2.0*self.sn)

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


class AEP_Model(object):

    def __init__(self, y_train):
        self.y_train = y_train
        self.N = y_train.shape[0]
        self.fixed_params = []
        self.updated = False

    def init_hypers(self, y_train, x_train=None):
        pass

    def get_hypers(self):
        pass

    def update_hypers(self, params):
        pass

    def optimise(
        self, method='L-BFGS-B', tol=None, reinit_hypers=True, 
        callback=None, maxfun=100000, maxiter=1000, alpha=0.5, mb_size=None, adam_lr=0.001, **kargs):
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
                results = adam(objective_wrapper, init_params_vec,
                               step_size=adam_lr,
                               maxiter=maxiter,
                               args=(params_args, self, mb_size, alpha))
                final_params = results
            else:
                options = {'maxfun': maxfun, 'maxiter': maxiter, 'disp': True, 'gtol': 1e-8}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, self.N, alpha),
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
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)

    def save_model(self, fname='/tmp/model.pickle'):
        params = self.get_hypers()
        pickle.dump(params, open(fname, "wb"))

    def load_model(self, fname='/tmp/model.pickle'):
        params = pickle.load(open(fname, "rb"))
        self.update_hypers(params)


class SGPLVM(AEP_Model):

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
        self.factor_x1 = np.zeros((N, Din))
        self.factor_x2 = np.zeros((N, Din))

        self.prior_x1 = prior_mean
        self.prior_x2 = prior_var

        self.x_post_1 = np.zeros((N, Din))
        self.x_post_2 = np.zeros((N, Din))

    @profile
    def objective_function(self, params, mb_size, alpha=1.0):
        N = self.N
        if mb_size == N:
            idxs = np.arange(N)
            yb = self.y_train
        else:
            idxs = np.random.randint(0, N, size=mb_size)
            yb = self.y_train[idxs, :]
        batch_size = yb.shape[0]
        scale_logZ = - N * 1.0 / batch_size / alpha
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        
        # update model with new hypers
        self.update_hypers(params)
        self.sgp_layer.compute_cavity(alpha)

        # compute cavity
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]
        cav_t1 = t01 + (1.0 - alpha) * t11
        cav_t2 = t02 + (1.0 - alpha) * t12
        vx = 1.0 / cav_t2
        mx = cav_t1 / cav_t2

        # propagate x cavity forward
        mout, vout, psi1, psi2 = self.sgp_layer.forward_prop_thru_cav(mx, vx)
        # compute logZ and gradients
        logZ, dm, dv = self.lik_layer.compute_log_Z(mout, vout, yb, alpha)
        logZ_scale = scale_logZ * logZ
        dm_scale = scale_logZ * dm
        dv_scale = scale_logZ * dv
        sgp_grad_hyper, sgp_grad_input = self.sgp_layer.backprop_grads_lvm(
            mout, vout, dm_scale, dv_scale, psi1, psi2, mx, vx, alpha)
        lik_grad_hyper = self.lik_layer.backprop_grads(
            mout, vout, dm, dv, alpha, scale_logZ)
        
        grad_all = {}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute grad wrt x params
        dcav_dt11 = (1.0-alpha) * cav_t1 / cav_t2
        dcav_dt12 = (1.0-alpha) * (-0.5*cav_t1**2/cav_t2**2 - 0.5/cav_t2)
        dmx = sgp_grad_input['mx']
        dvx = sgp_grad_input['vx']
        dlogZ_dt11 = (1.0-alpha) * dmx/cav_t2
        dlogZ_dt12 = (1.0-alpha) * (-dmx*cav_t1/cav_t2**2 - dvx/cav_t2**2)

        post_t1 = t01 + t11
        post_t2 = t02 + t12
        dpost_dt11 = post_t1 / post_t2
        dpost_dt12 = - 0.5 * post_t1**2 / post_t2**2 - 0.5 / post_t2

        scale_x = N * 1.0 / batch_size
        grad_all['x1'] = dlogZ_dt11 + scale_x * (
            - 1.0 / alpha * dcav_dt11 - (1.0 - 1.0 / alpha) * dpost_dt11)
        grad_all['x2'] = dlogZ_dt12 + scale_x * (
            - 1.0 / alpha * dcav_dt12 - (1.0 - 1.0 / alpha) * dpost_dt12)
        grad_all['x2'] *= 2 * t12

        # compute objective
        sgp_contrib = self.sgp_layer.compute_phi(alpha)

        phi_prior_x = self.compute_phi_prior_x(idxs)
        phi_poste_x = self.compute_phi_posterior_x(idxs)
        phi_cavity_x = self.compute_phi_cavity_x(idxs, alpha)
        x_contrib = scale_x * (
            phi_prior_x - 1.0/alpha*phi_cavity_x 
            - (1.0 - 1.0/alpha)*phi_poste_x)

        energy = logZ_scale + x_contrib + sgp_contrib

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def compute_phi_prior_x(self, idxs):
        t1 = self.prior_x1
        t2 = self.prior_x2
        m = t1 / t2
        v = 1.0 / t2
        Nb = idxs.shape[0]
        return 0.5 * Nb * self.Din * (m**2 / v + np.log(v))

    def compute_phi_posterior_x(self, idxs):
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]
        t1 = t01 + t11
        t2 = t02 + t12
        m = t1 / t2
        v = 1.0 / t2
        return np.sum(0.5 * (m**2 / v + np.log(v)))

    def compute_phi_cavity_x(self, idxs, alpha=1.0):
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]
        t1 = t01 + (1.0 - alpha) * t11
        t2 = t02 + (1.0 - alpha) * t12
        m = t1 / t2
        v = 1.0 / t2
        return np.sum(0.5 * (m**2 / v + np.log(v)))

    def predict_f(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def predict_y(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def get_posterior_x(self):
        post_1 = self.prior_x1 + self.factor_x1
        post_2 = self.prior_x2 + self.factor_x2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def impute_missing(self, y, missing_mask, alpha=0.5, no_iters=10, add_noise=False):
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
        reg.optimise(method='L-BFGS-B', alpha=0.5, maxiter=100)
        sgp_params = reg.sgp_layer.get_hypers()
        lik_params = self.lik_layer.init_hypers()
        init_params = dict(sgp_params)
        init_params.update(lik_params)
        init_params.update(x_params)
        return init_params

    def get_hypers(self):
        sgp_params = self.sgp_layer.get_hypers()
        lik_params = self.lik_layer.get_hypers()
        x_params = {}
        x_params['x1'] = self.factor_x1
        x_params['x2'] = np.log(self.factor_x2) / 2.0

        params = dict(sgp_params)
        params.update(lik_params)
        params.update(x_params)
        return params

    def update_hypers(self):
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)
        self.factor_x1 = params['x1']
        self.factor_x2 = np.exp(2 * params['x2'])


class SGPR(AEP_Model):

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

    @profile
    def objective_function(self, params, mb_size, alpha=1.0):
        N = self.N
        if mb_size >= N:
            xb = self.x_train
            yb = self.y_train
        else:
            idxs = np.random.randint(0, N, size=mb_size)
            xb = self.x_train[idxs, :]
            yb = self.y_train[idxs, :]
        batch_size = yb.shape[0]
        scale_logZ = - N * 1.0 / batch_size / alpha
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        
        # update model with new hypers
        self.update_hypers(params)
        self.sgp_layer.compute_cavity(alpha)
        
        # propagate x cavity forward
        mout, vout, kfu = self.sgp_layer.forward_prop_thru_cav(xb)
        # compute logZ and gradients
        logZ, dm, dv = self.lik_layer.compute_log_Z(mout, vout, yb, alpha)
        logZ_scale = scale_logZ * logZ
        dm_scale = scale_logZ * dm
        dv_scale = scale_logZ * dv
        sgp_grad_hyper = self.sgp_layer.backprop_grads_reg(
            mout, vout, dm_scale, dv_scale, kfu, xb, alpha)
        lik_grad_hyper = self.lik_layer.backprop_grads(
            mout, vout, dm, dv, alpha, scale_logZ)
        
        grad_all = {}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute objective
        sgp_contrib = self.sgp_layer.compute_phi(alpha)
        energy = logZ_scale + sgp_contrib

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def predict_f(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def sample_f(self, inputs, no_samples=1):
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
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def init_hypers(self, y_train):
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

    def update_hypers(self, params):
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)


class SDGPR(AEP_Model):

    def __init__(self, x_train, y_train, no_pseudos, hidden_sizes, lik='Gaussian'):
        super(SDGPR, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = x_train.shape[1]
        self.size = [Din] + hidden_sizes + [Dout]
        self.L = L = len(self.size) - 1
        if not isinstance(no_pseudos, (list, tuple)):
            self.Ms = Ms = [no_pseudos for i in range(L)]
        else:
            self.Ms = Ms = no_pseudos
        self.x_train = x_train

        self.sgp_layers = []
        for i in range(L):
            Din_i = self.size[i]
            Dout_i = self.size[i+1]
            M_i = self.Ms[i]
            self.sgp_layers.append(SGP_Layer(N, Din_i, Dout_i, M_i))

        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

    def objective_function(self, params, mb_size, alpha=1.0):
        N = self.N
        if mb_size >= N:
            xb = self.x_train
            yb = self.y_train
        else:
            idxs = np.random.randint(0, N, mb_size)
            xb = self.x_train[idxs, :]
            yb = self.y_train[idxs, :]
        batch_size = yb.shape[0]
        scale_logZ = - N * 1.0 / batch_size / alpha
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        
        # update model with new hypers
        self.update_hypers(params)
        for layer in self.sgp_layers:
            layer.compute_cavity(alpha)

        #### propagate x cavity forward
        mout, vout, psi1, psi2 = [], [], [], []
        for i in range(0, self.L):
            layer = self.sgp_layers[i]
            if i == 0:
                # first layer
                m0, v0, kfu0 = layer.forward_prop_thru_cav(xb)
                mout.append(m0)
                vout.append(v0)
                psi1.append(kfu0)
                psi2.append(None)
            else:
                mi, vi, psi1i, psi2i = layer.forward_prop_thru_cav(
                    mout[i-1], vout[i-1])
                mout.append(mi)
                vout.append(vi)
                psi1.append(psi1i)
                psi2.append(psi2i)

        # compute logZ and gradients
        logZ, dm, dv = self.lik_layer.compute_log_Z(
            mout[-1], vout[-1], yb, alpha)
        logZ_scale = scale_logZ * logZ
        dmi = scale_logZ * dm
        dvi = scale_logZ * dv
        grad_list = []
        for i in range(self.L-1, -1, -1):
            layer = self.sgp_layers[i]
            if i == 0:
                grad_hyper = layer.backprop_grads_reg(
                    mout[i], vout[i], dmi, dvi, psi1[i], xb, alpha)
            else:
                grad_hyper, grad_input = layer.backprop_grads_lvm(
                    mout[i], vout[i], dmi, dvi, psi1[i], psi2[i], 
                    mout[i-1], vout[i-1], alpha)
                dmi, dvi = grad_input['mx'], grad_input['vx']
            grad_list.insert(0, grad_hyper)

        lik_grad_hyper = self.lik_layer.backprop_grads(
            mout[-1], vout[-1], dm, dv, alpha, scale_logZ)
        
        grad_all = {}
        for i, grad in enumerate(grad_list):
            for key in grad.keys():
                grad_all[key + '_%d'%i] = grad[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute objective
        sgp_contrib = 0
        for layer in self.sgp_layers:
            sgp_contrib += layer.compute_phi(alpha)
        energy = logZ_scale + sgp_contrib

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def predict_f(self, inputs):
        if not self.updated:
            for layer in self.sgp_layers:
                layer.update_posterior_for_prediction()
            self.updated = True
        for i, layer in enumerate(self.sgp_layers):
            if i == 0:
                mf, vf = layer.forward_prop_thru_post(inputs)
            else:
                mf, vf = layer.forward_prop_thru_post(mf, vf)
        return mf, vf

    def sample_f(self, inputs, no_samples=1):
        if not self.updated:
            for layer in self.sgp_layers:
                layer.update_posterior_for_prediction()
            self.updated = True
        K = no_samples
        fs = np.zeros((inputs.shape[0], self.Dout, K))
        # TODO: remove for loop here
        for k in range(K):
            inputs_k = inputs
            for layer in self.sgp_layers:
                outputs = layer.sample(inputs_k)
                inputs_k = outputs
            fs[:, :, k] = outputs
        return fs

    def predict_y(self, inputs):
        mf, vf = self.predict_f(inputs)
        my, vy = self.lik_layer.forward_prop_thru_post(mf, vf)
        return my, vy

    def init_hypers(self, y_train):
        init_params = dict()
        for i in range(self.L):
            if i == 0:
                sgp_params = self.sgp_layers[i].init_hypers(
                    self.x_train, 
                    key_suffix='_%d'%i)
            else:
                sgp_params = self.sgp_layers[i].init_hypers(
                    key_suffix='_%d'%i)
            init_params.update(sgp_params)

        lik_params = self.lik_layer.init_hypers()
        init_params.update(lik_params)
        return init_params

    def get_hypers(self):
        params = dict()
        for i in range(self.L):
            sgp_params = self.sgp_layers[i].get_hypers(key_suffix='_%d'%i)
            params.update(sgp_params)
        lik_params = self.lik_layer.get_hypers()
        params.update(lik_params)
        return params

    def update_hypers(self, params):
        for i, layer in enumerate(self.sgp_layers):
            layer.update_hypers(params, key_suffix='_%d'%i)
        self.lik_layer.update_hypers(params)


class Gauss_Emis():

    def __init__(self, y, Dout, Din):
        self.y = y
        self.N = y.shape[0]
        self.Dout = Dout
        self.Din = Din
        self.C = np.zeros((Dout, Din))
        self.R = np.zeros(Dout)

    def update_hypers(self, params):
        self.C = params['C']
        self.R = np.exp(2*params['R'])

    def init_hypers(self):
        params = {}
        params['C'] = np.ones((self.Dout, self.Din)) / (self.Dout*self.Din)
        params['R'] = np.log(0.01) * np.ones(self.Dout)
        return params

    def get_hypers(self):
        params = {'C': self.C, 'R': 0.5 * np.log(self.R)}
        return params

    def compute_factor(self, x_cav_m, x_cav_v, alpha):
        # this update does not depend on the cavity information and alpha
        CTRinv = self.C.T / self.R
        t2 = np.sum(CTRinv * self.C.T, axis=1)
        t1 = np.dot(CTRinv, self.y.T)
        t2rep = np.tile(t2[np.newaxis, :], [self.N, 1])
        t1 = t1.T
        return t1, t2rep

    def output_probabilistic(self, mf, vf):
        my = np.einsum('ab,nb->na', self.C, mf)
        vy_noiseless = np.einsum('ab,nb,bc->nac', self.C, vf, self.C.T)
        vy = vy_noiseless + np.diag(self.R)
        return my, vy_noiseless, vy

    def compute_emission_tilted(self, mx, vx, alpha, scale):
        C = self.C
        R = self.R
        N = self.N
        Dout = self.Dout
        Vy = np.diag(R/alpha) + np.einsum('da,na,ab->ndb', C, vx, C.T)
        Ly = np.linalg.cholesky(Vy)
        Ydiff = self.y - np.einsum('da,na->nd', C, mx)

        VinvY = np.linalg.solve(Vy, Ydiff)
        quad_term = -0.5 * np.sum(Ydiff * VinvY)
        Vlogdet_term = - np.sum(np.log(np.diagonal(Ly, axis1=1, axis2=2)))
        Rlogdet_term = 0.5 * N * (1-alpha) * np.sum(np.log(R))
        const_term = - N * Dout * (0.5*alpha*np.log(2*np.pi) + np.log(alpha))
        logZ = const_term + Rlogdet_term + Vlogdet_term + quad_term

        Vyinv = np.linalg.inv(Vy)
        dR = (-0.5 * np.sum(np.diagonal(Vyinv, axis1=1, axis2=2), axis=0) 
            + 0.5 * np.sum(VinvY**2, axis=0)) / alpha
        dR += 0.5 * N * (1-alpha) / R
        dR *= 2*R

        dSigma = -0.5 * Vyinv + 0.5 * np.einsum('na,nb->nab', VinvY, VinvY)
        dmu = VinvY
        dC1 = np.einsum('na,nb->ab', dmu, mx) 
        dC2 = 2*np.einsum('nc,bc,nab->ac', vx, C, dSigma)
        dC = dC1 + dC2

        dmx = np.einsum('na,ab->nb', dmu, C)
        dvx = np.einsum('nab,da,db->nd', dSigma, C.T, C.T)

        emi_grads = {'C': dC*scale, 'R': dR*scale}
        input_grads = {'mx': dmx*scale, 'vx': dvx*scale}
        return logZ*scale, input_grads, emi_grads


class SGPSSM_old_working(AEP_Model):

    def __init__(self, y_train, hidden_size, no_pseudo, 
        lik='Gaussian', prior_mean=0, prior_var=1):
        super(SGPSSM, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = hidden_size
        self.M = M = no_pseudo

        self.sgp_layer = SGP_Layer(N-1, Din, Din, M)
        if lik.lower() == 'gaussian':
            self.emi_layer = Gauss_Emis(y_train, Dout, Din)
        elif lik.lower() == 'probit':
            self.emi_layer = Probit_Emis(y_train, Dout, Din)
        else:
            raise NotImplementedError('likelihood not implemented')

        # natural params for latent variables
        self.x_factor_1 = np.zeros((N, Din))
        self.x_factor_2 = np.zeros((N, Din))
        self.x_prior_1 = prior_mean / prior_var
        self.x_prior_2 = 1.0 / prior_var

        self.x_post_1 = np.zeros((N, Din))
        self.x_post_2 = np.zeros((N, Din))

    @profile
    def objective_function(self, params, mb_size, alpha=1.0):
        # self.plot()
        N = self.N
        # TODO: deal with minibatch here
        batch_size_dyn = (N-1)
        scale_logZ_dyn = - 1.0 / alpha
        batch_size_emi = N
        scale_logZ_emi = - 1.0 / alpha
        
        # update model with new hypers
        self.update_hypers(params)
        self.sgp_layer.compute_cavity(alpha)
        cav_m, cav_v, cav_1, cav_2 = self.compute_cavity_x(alpha)
        # deal with the transition/dynamic factors here
        idxs_prev = np.arange(1, self.N)
        cav_t_m, cav_t_v = cav_m[idxs_prev, :], cav_v[idxs_prev, :]
        idxs_next = np.arange(0, self.N-1)
        cav_tm1_m, cav_tm1_v = cav_m[idxs_next, :], cav_v[idxs_next, :]
        mprop, vprop, psi1, psi2 = self.sgp_layer.forward_prop_thru_cav(
            cav_tm1_m, cav_tm1_v)
        logZ_dyn, dmprop, dvprop, dmt, dvt, dsn = self.compute_transition_tilted(
            mprop, vprop, cav_t_m, cav_t_v, alpha, scale_logZ_dyn)
        sgp_grad_hyper, sgp_grad_input = self.sgp_layer.backprop_grads_lvm(
            mprop, vprop, dmprop, dvprop, 
            psi1, psi2, cav_tm1_m, cav_tm1_v, alpha)

        # deal with the emission factors here
        idxs_up = np.arange(0, self.N)
        cav_up_m, cav_up_v = cav_m[idxs_up, :], cav_v[idxs_up, :]
        logZ_emi, input_grads, emi_grads = self.emi_layer.compute_emission_tilted(
            cav_up_m, cav_up_v, alpha, scale_logZ_emi)

        # collect emission and GP hyperparameters
        grad_all = {'sn': dsn}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]
        for key in emi_grads.keys():
            grad_all[key] = emi_grads[key]

        dmcav_up, dvcav_up = input_grads['mx'], input_grads['vx']
        dmcav_prev, dvcav_prev = dmt, dvt
        dmcav_next, dvcav_next = sgp_grad_input['mx'], sgp_grad_input['vx']
        
        # compute posterior
        grads_x_via_post = self.compute_posterior_grad_x(alpha)
        grads_x_via_cavity = self.compute_cavity_grad_x(
            alpha,
            cav_1, cav_2)
        grads_x_via_logZ = self.compute_logZ_grad_x(
            alpha, cav_1, cav_2, dmcav_up, dvcav_up,
            dmcav_prev, dvcav_prev, dmcav_next, dvcav_next)

        grads_x = {}
        for key in ['x_factor_1', 'x_factor_2']:
            grads_x[key] = grads_x_via_post[key] + grads_x_via_cavity[key] + grads_x_via_logZ[key]

        for key in grads_x.keys():
            grad_all[key] = grads_x[key]

        # compute objective
        sgp_contrib = self.sgp_layer.compute_phi(alpha)
        phi_prior_x = self.compute_phi_prior_x()
        phi_poste_x = self.compute_phi_posterior_x(alpha)
        phi_cavity_x = self.compute_phi_cavity_x(alpha)
        x_contrib = phi_prior_x + phi_poste_x + phi_cavity_x
        energy = logZ_dyn + logZ_emi + x_contrib + sgp_contrib
        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        # energy /= self.N
        # for key in grad_all.keys():
        #     grad_all[key] /= self.N
        
        return energy, grad_all

    def compute_posterior_grad_x(self, alpha):
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        dpost_1 = post_1 / post_2
        dpost_2 = - 0.5 * post_1**2 / post_2**2 - 0.5 / post_2
        scale_x_post = - (1.0 - 1.0 / alpha) * np.ones((self.N, 1))
        scale_x_post[0:self.N-1] = scale_x_post[0:self.N-1] + 1.0/alpha
        scale_x_post[1:self.N] = scale_x_post[1:self.N] + 1.0/alpha
        grads_x_1 = scale_x_post * dpost_1
        grads_x_2 = scale_x_post * dpost_2
        grads = {}
        grads['x_factor_1'] = 3.0*grads_x_1
        grads['x_factor_2'] = 6.0*grads_x_2*self.x_factor_2
        grads['x_factor_1'][[0, -1], :] = 2.0*grads_x_1[[0, -1], :]
        grads['x_factor_2'][[0, -1], :] = (
            4.0*grads_x_2[[0, -1], :] * self.x_factor_2[[0, -1], :])
        return grads

    def compute_logZ_grad_x(self, alpha, cav_1, cav_2, dmcav_up, dvcav_up, 
        dmcav_prev, dvcav_prev, dmcav_next, dvcav_next):

        grads_up_1 = dmcav_up / cav_2
        grads_up_2 = - dmcav_up * cav_1 / cav_2**2 - dvcav_up / cav_2**2
        grads_x_1 = grads_up_1
        grads_x_2 = grads_up_2

        idxs = np.arange(1, self.N)
        grads_prev_1 = dmcav_prev / cav_2[idxs, :]
        grads_prev_2 = (- dmcav_prev * cav_1[idxs, :] / cav_2[idxs, :]**2 
            - dvcav_prev / cav_2[idxs, :]**2)
        grads_x_1[idxs, :] += grads_prev_1
        grads_x_2[idxs, :] += grads_prev_2

        idxs = np.arange(0, self.N-1)
        grads_next_1 = dmcav_next / cav_2[idxs, :]
        grads_next_2 = (- dmcav_next * cav_1[idxs, :] / cav_2[idxs, :]**2 
            - dvcav_next / cav_2[idxs, :]**2)
        grads_x_1[idxs, :] += grads_next_1
        grads_x_2[idxs, :] += grads_next_2

        scale_x_cav = (3.0 - alpha)*np.ones((self.N, 1))
        scale_x_cav[0] = 2.0 - alpha
        scale_x_cav[-1] = 2.0 - alpha
        grad_1 = grads_x_1 * scale_x_cav
        grad_2 = grads_x_2 * scale_x_cav
        grads = {}
        grads['x_factor_1'] = grad_1
        grads['x_factor_2'] = grad_2 * 2 * self.x_factor_2
        return grads

    def compute_cavity_grad_x(self, alpha, cav_1, cav_2):
        scale = -1.0/alpha
        dcav_1 = cav_1 / cav_2
        dcav_2 = - 0.5 * cav_1**2 / cav_2**2 - 0.5 / cav_2
        scale_x_cav = scale * np.ones((self.N, 1))
        scale_x_cav[0:self.N-1] = scale_x_cav[0:self.N-1] + scale
        scale_x_cav[1:self.N] = scale_x_cav[1:self.N] + scale
        grads_x_1 = scale_x_cav * dcav_1
        grads_x_2 = scale_x_cav * dcav_2

        scale_x_cav = (3.0 - alpha)*np.ones((self.N, 1))
        scale_x_cav[0] = 2.0 - alpha
        scale_x_cav[-1] = 2.0 - alpha
        grad_1 = grads_x_1 * scale_x_cav
        grad_2 = grads_x_2 * scale_x_cav
        grads = {}
        grads['x_factor_1'] = grad_1
        grads['x_factor_2'] = grad_2 * 2 * self.x_factor_2
        return grads

    def compute_transition_tilted(self, m_prop, v_prop, m_t, v_t, alpha, scale):
        sn2 = np.exp(2*self.sn)
        v_sum = v_t + v_prop + sn2 / alpha
        m_diff = m_t - m_prop
        exp_term = -0.5 * m_diff**2 / v_sum
        const_term = -0.5 * np.log(2*np.pi*v_sum)
        alpha_term = 0.5 * (1-alpha) * np.log(2*np.pi*sn2) - 0.5*np.log(alpha)
        logZ = exp_term + const_term + alpha_term
        logZ = np.sum(logZ)

        dvt = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dvprop = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dmt = -m_diff / v_sum
        dmprop = m_diff / v_sum

        dv_sum = np.sum(dvt)
        dsn = dv_sum*2*sn2/alpha + m_prop.shape[0]*self.Din*(1-alpha)
        return scale*logZ, scale*dmprop, scale*dvprop, scale*dmt, scale*dvt, scale*dsn

    def compute_cavity_x(self, alpha):
        cav_1 = self.x_post_1 - alpha*self.x_factor_1
        cav_2 = self.x_post_2 - alpha*self.x_factor_2
        return cav_1/(cav_2+1e-16), 1.0/(cav_2+1e-16), cav_1, cav_2

    def compute_phi_prior_x(self):
        t1 = self.x_prior_1
        t2 = self.x_prior_2
        m = t1 / t2
        v = 1.0 / t2
        return 0.5 * self.Din * (m**2 / v + np.log(v))

    def compute_phi_posterior_x(self, alpha):
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        phi_post = 0.5 * (post_1**2 / post_2 - np.log(post_2))
        scale_x_post = - (1.0 - 1.0 / alpha) * np.ones((self.N, 1))
        scale_x_post[0:self.N-1] = scale_x_post[0:self.N-1] + 1/alpha
        scale_x_post[1:self.N] = scale_x_post[1:self.N] + 1/alpha
        return np.sum(scale_x_post * phi_post)

    def compute_phi_cavity_x(self, alpha):
        scale = -1.0/alpha
        cav_1 = self.x_post_1 - alpha*self.x_factor_1
        cav_2 = self.x_post_2 - alpha*self.x_factor_2
        phi_cav = 0.5 * (cav_1**2 / cav_2 - np.log(cav_2))

        scale_x_cav = scale * np.ones((self.N, 1))
        scale_x_cav[0:self.N-1] = scale_x_cav[0:self.N-1] + scale
        scale_x_cav[1:self.N] = scale_x_cav[1:self.N] + scale
        phi_cav_sum = np.sum(scale_x_cav * phi_cav)
        return phi_cav_sum

    def predict_f(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def predict_y(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.emi_layer.output_probabilistic(mf, vf)
        return my, vy

    def get_posterior_x(self):
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def init_hypers_old(self, y_train):
        # TODO: alternatitve method for non real-valued data
        if self.Din == self.Dout:
            post_m = np.copy(y_train)
            post_v = 0.01 * np.ones_like(post_m)
        else:
            post_m = PCA_reduce(y_train, self.Din)
            post_m_mean = np.mean(post_m, axis=0)
            post_m_std = np.std(post_m, axis=0)
            post_m = (post_m - post_m_mean) / post_m_std
            post_v = 0.01 * np.ones_like(post_m)

        sgp_params = self.sgp_layer.init_hypers()
        emi_params = self.emi_layer.init_hypers()
        
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        ssm_params = {'sn': np.log(0.01)}
        ssm_params['x_factor_1'] = post_1/3
        ssm_params['x_factor_2'] = np.log(post_2/3) / 2
        init_params = dict(sgp_params)
        init_params.update(emi_params)
        init_params.update(ssm_params)
        return init_params

    def init_hypers(self, y_train):
        # initialise latent variables and emission using a Gaussian LDS
        print 'init latent variable using LDS...'
        from pylds.models import DefaultLDS
        model = DefaultLDS(self.Dout, self.Din, 0)
        model.add_data(y_train)
        # Initialize with a few iterations of Gibbs
        for _ in range(100):
            model.resample_model()
        # run EM
        for _ in range(50):
            model.EM_step()

        s = model.states_list.pop()    
        s.info_E_step()
        post_m = s.smoothed_mus
        # post_v = np.diagonal(s.smoothed_sigmas, axis1=1, axis2=2)
        scale = np.std(post_m, axis=0)
        post_m = (post_m - np.mean(post_m, axis=0)) / scale
        # post_v = post_v / scale**2
        post_v = 0.01*np.ones_like(post_m)
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        ssm_params = {'sn': np.log(0.01)}
        ssm_params['x_factor_1'] = post_1/3
        ssm_params['x_factor_2'] = np.log(post_2/3) / 2
        emi_params = self.emi_layer.init_hypers()
        if isinstance(self.emi_layer, Gauss_Emis):
            # emi_params['C'] = np.dot(s.C, np.diag(scale))
            # emi_params['C'] = s.C
            emi_params['C'] = np.dot(s.C, np.diag(scale))*np.random.randn(self.Dout, self.Din)
            

        # learnt a GP mapping between hidden states
        print 'init latent function using GPR...'
        reg = SGPR(post_m[:self.N-1, :], post_m[1:, :], self.M, 'Gaussian')
        reg.set_fixed_params(['sn', 'sf', 'ls', 'zu'])
        reg.optimise(method='L-BFGS-B', alpha=0.5, maxiter=100)
        sgp_params = reg.sgp_layer.get_hypers()
        # sgp_params['ls'] -= np.log(5)
        # sgp_params = self.sgp_layer.init_hypers(post_m)
        # sgp_params['zu'] = post_m[np.random.randint(0, post_m.shape[0], self.M), :]
        # sgp_params['ls'] = np.log(1.0)*np.ones(self.Din)
        init_params = dict(sgp_params)
        init_params.update(emi_params)
        init_params.update(ssm_params)
        return init_params

    def get_hypers(self):
        sgp_params = self.sgp_layer.get_hypers()
        emi_params = self.emi_layer.get_hypers()
        ssm_params = {}
        ssm_params['x_factor_1'] = self.x_factor_1
        ssm_params['x_factor_2'] = np.log(self.x_factor_2) / 2.0
        ssm_params['sn'] = self.sn
        params = dict(sgp_params)
        params.update(emi_params)
        params.update(ssm_params)
        return params

    def update_hypers(self, params):
        self.sgp_layer.update_hypers(params)
        self.emi_layer.update_hypers(params)
        self.sn = params['sn']
        self.x_factor_1 = params['x_factor_1']
        self.x_factor_2 = np.exp(2*params['x_factor_2'])

        self.x_post_1 = 3*self.x_factor_1
        self.x_post_2 = 3*self.x_factor_2
        self.x_post_1[[0, -1], :] = 2*self.x_factor_1[[0, -1], :]
        self.x_post_2[[0, -1], :] = 2*self.x_factor_2[[0, -1], :]
        self.x_post_1[0, :] += self.x_prior_1
        self.x_post_2[0, :] += self.x_prior_2


# with control, in progress, TODO: prediction
class SGPSSM(AEP_Model):

    def __init__(self, y_train, hidden_size, no_pseudo, 
        lik='Gaussian', prior_mean=0, prior_var=1, x_control=None):
        super(SGPSSM, self).__init__(y_train)
        if x_control:
            self.Dcon = Dcon = x_control.shape[1]
            self.x_control = x_control
        else:
            self.Dcon = Dcon = 0
            self.x_control = None
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = hidden_size
        self.M = M = no_pseudo

        self.sgp_layer = SGP_Layer(N-1, Din+Dcon, Din, M)
        if lik.lower() == 'gaussian':
            self.emi_layer = Gauss_Emis(y_train, Dout, Din)
        elif lik.lower() == 'probit':
            self.emi_layer = Probit_Emis(y_train, Dout, Din)
        else:
            raise NotImplementedError('likelihood not implemented')

        # natural params for latent variables
        self.x_factor_1 = np.zeros((N, Din))
        self.x_factor_2 = np.zeros((N, Din))
        self.x_prior_1 = prior_mean / prior_var
        self.x_prior_2 = 1.0 / prior_var

        self.x_post_1 = np.zeros((N, Din))
        self.x_post_2 = np.zeros((N, Din))

    @profile
    def objective_function(self, params, mb_size, alpha=1.0):
        # self.plot()
        N = self.N
        # TODO: deal with minibatch here
        batch_size_dyn = (N-1)
        scale_logZ_dyn = - 1.0 / alpha
        batch_size_emi = N
        scale_logZ_emi = - 1.0 / alpha
        
        # update model with new hypers
        self.update_hypers(params)
        self.sgp_layer.compute_cavity(alpha)
        cav_m, cav_v, cav_1, cav_2 = self.compute_cavity_x(alpha)
        # deal with the transition/dynamic factors here
        idxs_prev = np.arange(1, self.N)
        cav_t_m, cav_t_v = cav_m[idxs_prev, :], cav_v[idxs_prev, :]
        idxs_next = np.arange(0, self.N-1)
        cav_tm1_m, cav_tm1_v = cav_m[idxs_next, :], cav_v[idxs_next, :]
        if self.Dcon > 0:
            cav_tm1_mc = np.hstack((cav_tm1_m, self.x_control[idxs_next, :]))
            cav_tm1_vc = np.hstack((cav_tm1_v, np.zeros((self.N-1, self.Dcon))))
        else:
            cav_tm1_mc, cav_tm1_vc = cav_tm1_m, cav_tm1_v
        mprop, vprop, psi1, psi2 = self.sgp_layer.forward_prop_thru_cav(
            cav_tm1_mc, cav_tm1_vc)
        logZ_dyn, dmprop, dvprop, dmt, dvt, dsn = self.compute_transition_tilted(
            mprop, vprop, cav_t_m, cav_t_v, alpha, scale_logZ_dyn)
        sgp_grad_hyper, sgp_grad_input = self.sgp_layer.backprop_grads_lvm(
            mprop, vprop, dmprop, dvprop, 
            psi1, psi2, cav_tm1_mc, cav_tm1_vc, alpha)

        # deal with the emission factors here
        idxs_up = np.arange(0, self.N)
        cav_up_m, cav_up_v = cav_m[idxs_up, :], cav_v[idxs_up, :]
        logZ_emi, input_grads, emi_grads = self.emi_layer.compute_emission_tilted(
            cav_up_m, cav_up_v, alpha, scale_logZ_emi)

        # collect emission and GP hyperparameters
        grad_all = {'sn': dsn}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]
        for key in emi_grads.keys():
            grad_all[key] = emi_grads[key]

        dmcav_up, dvcav_up = input_grads['mx'], input_grads['vx']
        dmcav_prev, dvcav_prev = dmt, dvt
        dmcav_next = sgp_grad_input['mx'][:, :self.Din]
        dvcav_next = sgp_grad_input['vx'][:, :self.Din]
        
        # compute posterior
        grads_x_via_post = self.compute_posterior_grad_x(alpha)
        grads_x_via_cavity = self.compute_cavity_grad_x(
            alpha,
            cav_1, cav_2)
        grads_x_via_logZ = self.compute_logZ_grad_x(
            alpha, cav_1, cav_2, dmcav_up, dvcav_up,
            dmcav_prev, dvcav_prev, dmcav_next, dvcav_next)

        grads_x = {}
        for key in ['x_factor_1', 'x_factor_2']:
            grads_x[key] = grads_x_via_post[key] + grads_x_via_cavity[key] + grads_x_via_logZ[key]

        for key in grads_x.keys():
            grad_all[key] = grads_x[key]

        # compute objective
        sgp_contrib = self.sgp_layer.compute_phi(alpha)
        phi_prior_x = self.compute_phi_prior_x()
        phi_poste_x = self.compute_phi_posterior_x(alpha)
        phi_cavity_x = self.compute_phi_cavity_x(alpha)
        x_contrib = phi_prior_x + phi_poste_x + phi_cavity_x
        energy = logZ_dyn + logZ_emi + x_contrib + sgp_contrib
        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        # energy /= self.N
        # for key in grad_all.keys():
        #     grad_all[key] /= self.N
        
        return energy, grad_all

    def compute_posterior_grad_x(self, alpha):
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        dpost_1 = post_1 / post_2
        dpost_2 = - 0.5 * post_1**2 / post_2**2 - 0.5 / post_2
        scale_x_post = - (1.0 - 1.0 / alpha) * np.ones((self.N, 1))
        scale_x_post[0:self.N-1] = scale_x_post[0:self.N-1] + 1.0/alpha
        scale_x_post[1:self.N] = scale_x_post[1:self.N] + 1.0/alpha
        grads_x_1 = scale_x_post * dpost_1
        grads_x_2 = scale_x_post * dpost_2
        grads = {}
        grads['x_factor_1'] = 3.0*grads_x_1
        grads['x_factor_2'] = 6.0*grads_x_2*self.x_factor_2
        grads['x_factor_1'][[0, -1], :] = 2.0*grads_x_1[[0, -1], :]
        grads['x_factor_2'][[0, -1], :] = (
            4.0*grads_x_2[[0, -1], :] * self.x_factor_2[[0, -1], :])
        return grads

    def compute_logZ_grad_x(self, alpha, cav_1, cav_2, dmcav_up, dvcav_up, 
        dmcav_prev, dvcav_prev, dmcav_next, dvcav_next):

        grads_up_1 = dmcav_up / cav_2
        grads_up_2 = - dmcav_up * cav_1 / cav_2**2 - dvcav_up / cav_2**2
        grads_x_1 = grads_up_1
        grads_x_2 = grads_up_2

        idxs = np.arange(1, self.N)
        grads_prev_1 = dmcav_prev / cav_2[idxs, :]
        grads_prev_2 = (- dmcav_prev * cav_1[idxs, :] / cav_2[idxs, :]**2 
            - dvcav_prev / cav_2[idxs, :]**2)
        grads_x_1[idxs, :] += grads_prev_1
        grads_x_2[idxs, :] += grads_prev_2

        idxs = np.arange(0, self.N-1)
        grads_next_1 = dmcav_next / cav_2[idxs, :]
        grads_next_2 = (- dmcav_next * cav_1[idxs, :] / cav_2[idxs, :]**2 
            - dvcav_next / cav_2[idxs, :]**2)
        grads_x_1[idxs, :] += grads_next_1
        grads_x_2[idxs, :] += grads_next_2

        scale_x_cav = (3.0 - alpha)*np.ones((self.N, 1))
        scale_x_cav[0] = 2.0 - alpha
        scale_x_cav[-1] = 2.0 - alpha
        grad_1 = grads_x_1 * scale_x_cav
        grad_2 = grads_x_2 * scale_x_cav
        grads = {}
        grads['x_factor_1'] = grad_1
        grads['x_factor_2'] = grad_2 * 2 * self.x_factor_2
        return grads

    def compute_cavity_grad_x(self, alpha, cav_1, cav_2):
        scale = -1.0/alpha
        dcav_1 = cav_1 / cav_2
        dcav_2 = - 0.5 * cav_1**2 / cav_2**2 - 0.5 / cav_2
        scale_x_cav = scale * np.ones((self.N, 1))
        scale_x_cav[0:self.N-1] = scale_x_cav[0:self.N-1] + scale
        scale_x_cav[1:self.N] = scale_x_cav[1:self.N] + scale
        grads_x_1 = scale_x_cav * dcav_1
        grads_x_2 = scale_x_cav * dcav_2

        scale_x_cav = (3.0 - alpha)*np.ones((self.N, 1))
        scale_x_cav[0] = 2.0 - alpha
        scale_x_cav[-1] = 2.0 - alpha
        grad_1 = grads_x_1 * scale_x_cav
        grad_2 = grads_x_2 * scale_x_cav
        grads = {}
        grads['x_factor_1'] = grad_1
        grads['x_factor_2'] = grad_2 * 2 * self.x_factor_2
        return grads

    def compute_transition_tilted(self, m_prop, v_prop, m_t, v_t, alpha, scale):
        sn2 = np.exp(2*self.sn)
        v_sum = v_t + v_prop + sn2 / alpha
        m_diff = m_t - m_prop
        exp_term = -0.5 * m_diff**2 / v_sum
        const_term = -0.5 * np.log(2*np.pi*v_sum)
        alpha_term = 0.5 * (1-alpha) * np.log(2*np.pi*sn2) - 0.5*np.log(alpha)
        logZ = exp_term + const_term + alpha_term
        logZ = np.sum(logZ)

        dvt = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dvprop = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dmt = -m_diff / v_sum
        dmprop = m_diff / v_sum

        dv_sum = np.sum(dvt)
        dsn = dv_sum*2*sn2/alpha + m_prop.shape[0]*self.Din*(1-alpha)
        return scale*logZ, scale*dmprop, scale*dvprop, scale*dmt, scale*dvt, scale*dsn

    def compute_cavity_x(self, alpha):
        cav_1 = self.x_post_1 - alpha*self.x_factor_1
        cav_2 = self.x_post_2 - alpha*self.x_factor_2
        return cav_1/(cav_2+1e-16), 1.0/(cav_2+1e-16), cav_1, cav_2

    def compute_phi_prior_x(self):
        t1 = self.x_prior_1
        t2 = self.x_prior_2
        m = t1 / t2
        v = 1.0 / t2
        return 0.5 * self.Din * (m**2 / v + np.log(v))

    def compute_phi_posterior_x(self, alpha):
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        phi_post = 0.5 * (post_1**2 / post_2 - np.log(post_2))
        scale_x_post = - (1.0 - 1.0 / alpha) * np.ones((self.N, 1))
        scale_x_post[0:self.N-1] = scale_x_post[0:self.N-1] + 1/alpha
        scale_x_post[1:self.N] = scale_x_post[1:self.N] + 1/alpha
        return np.sum(scale_x_post * phi_post)

    def compute_phi_cavity_x(self, alpha):
        scale = -1.0/alpha
        cav_1 = self.x_post_1 - alpha*self.x_factor_1
        cav_2 = self.x_post_2 - alpha*self.x_factor_2
        phi_cav = 0.5 * (cav_1**2 / cav_2 - np.log(cav_2))

        scale_x_cav = scale * np.ones((self.N, 1))
        scale_x_cav[0:self.N-1] = scale_x_cav[0:self.N-1] + scale
        scale_x_cav[1:self.N] = scale_x_cav[1:self.N] + scale
        phi_cav_sum = np.sum(scale_x_cav * phi_cav)
        return phi_cav_sum

    def predict_f(self, inputs):
        # TODO
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def predict_forward(self, T, x_control=None):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mx = np.zeros((T, self.Din))
        vx = np.zeros((T, self.Din))
        my = np.zeros((T, self.Dout))
        vy_noiseless = np.zeros((T, self.Dout, self.Dout))
        vy = np.zeros((T, self.Dout, self.Dout))
        post_m, post_v = self.get_posterior_x()
        mtm1 = post_m[[-1], :]
        vtm1 = post_v[[-1], :]
        for t in range(T):
            if x_control is not None:
                mtm1 = np.hstack((mtm1, x_control[[t], :]))
                vtm1 = np.hstack((vtm1, np.zeros((1, self.Dcon))))
            mt, vt = self.sgp_layer.forward_prop_thru_post(mtm1, vtm1)
            myt, vyt, vyt_n = self.emi_layer.output_probabilistic(mt, vt)
            mx[t, :], vx[t, :] = mt, vt
            my[t, :], vy_noiseless[t, :, :], vy[t, :, :] = myt, vyt, vyt_n
            mtm1 = mt
            vtm1 = vt
        return mx, vx, my, vy_noiseless, vy

    def predict_y(self, inputs):
        # TODO
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
        my, vy = self.emi_layer.output_probabilistic(mf, vf)
        return my, vy

    def get_posterior_x(self):
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def get_posterior_y(self):
        mx, vx = self.get_posterior_x()
        my, vy, vyn = self.emi_layer.output_probabilistic(mx, vx)
        return my, vy, vyn

    def init_hypers_old(self, y_train):
        # TODO: alternatitve method for non real-valued data
        if self.Din == self.Dout:
            post_m = np.copy(y_train)
            post_v = 0.01 * np.ones_like(post_m)
        else:
            post_m = PCA_reduce(y_train, self.Din)
            post_m_mean = np.mean(post_m, axis=0)
            post_m_std = np.std(post_m, axis=0)
            post_m = (post_m - post_m_mean) / post_m_std
            post_v = 0.01 * np.ones_like(post_m)

        sgp_params = self.sgp_layer.init_hypers()
        emi_params = self.emi_layer.init_hypers()
        
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        ssm_params = {'sn': np.log(0.01)}
        ssm_params['x_factor_1'] = post_1/3
        ssm_params['x_factor_2'] = np.log(post_2/3) / 2
        init_params = dict(sgp_params)
        init_params.update(emi_params)
        init_params.update(ssm_params)
        return init_params

    def init_hypers(self, y_train):
        # initialise latent variables and emission using a Gaussian LDS
        print 'init latent variable using LDS...'
        from pylds.models import DefaultLDS
        model = DefaultLDS(self.Dout, self.Din, self.Dcon)
        model.add_data(y_train, inputs=self.x_control)
        # Initialize with a few iterations of Gibbs
        for _ in range(100):
            model.resample_model()
        # run EM
        for _ in range(100):
            model.EM_step()

        s = model.states_list.pop()    
        s.info_E_step()
        post_m = s.smoothed_mus
        # post_v = np.diagonal(s.smoothed_sigmas, axis1=1, axis2=2)
        scale = np.std(post_m, axis=0)
        post_m = (post_m - np.mean(post_m, axis=0)) / scale
        # post_v = post_v / scale**2
        post_v = 0.01*np.ones_like(post_m)
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        ssm_params = {'sn': np.log(0.01)}
        ssm_params['x_factor_1'] = post_1/3
        ssm_params['x_factor_2'] = np.log(post_2/3) / 2
        emi_params = self.emi_layer.init_hypers()
        if isinstance(self.emi_layer, Gauss_Emis):
            # emi_params['C'] = np.dot(s.C, np.diag(scale))
            # emi_params['C'] = s.C
            emi_params['C'] = np.dot(s.C, np.diag(scale))*np.random.randn(self.Dout, self.Din)
            

        # learnt a GP mapping between hidden states
        print 'init latent function using GPR...'
        x = post_m[:self.N-1, :]
        y = post_m[1:, :]
        if self.Dcon > 0:
            x = np.hstack((x, self.x_control[:self.N-1, :]))
        reg = SGPR(x, y, self.M, 'Gaussian')
        reg.set_fixed_params(['sn', 'sf', 'ls', 'zu'])
        reg.optimise(method='L-BFGS-B', alpha=0.4, maxiter=150)
        sgp_params = reg.sgp_layer.get_hypers()
        sgp_params['ls'] -= np.log(5)
        # sgp_params = self.sgp_layer.init_hypers(post_m)
        # sgp_params['zu'] = post_m[np.random.randint(0, post_m.shape[0], self.M), :]
        # sgp_params['ls'] = np.log(1.0)*np.ones(self.Din)
        init_params = dict(sgp_params)
        init_params.update(emi_params)
        init_params.update(ssm_params)
        return init_params

    def get_hypers(self):
        sgp_params = self.sgp_layer.get_hypers()
        emi_params = self.emi_layer.get_hypers()
        ssm_params = {}
        ssm_params['x_factor_1'] = self.x_factor_1
        ssm_params['x_factor_2'] = np.log(self.x_factor_2) / 2.0
        ssm_params['sn'] = self.sn
        params = dict(sgp_params)
        params.update(emi_params)
        params.update(ssm_params)
        return params

    def update_hypers(self, params):
        self.sgp_layer.update_hypers(params)
        self.emi_layer.update_hypers(params)
        self.sn = params['sn']
        self.x_factor_1 = params['x_factor_1']
        self.x_factor_2 = np.exp(2*params['x_factor_2'])

        self.x_post_1 = 3*self.x_factor_1
        self.x_post_2 = 3*self.x_factor_2
        self.x_post_1[[0, -1], :] = 2*self.x_factor_1[[0, -1], :]
        self.x_post_2[[0, -1], :] = 2*self.x_factor_2[[0, -1], :]
        self.x_post_1[0, :] += self.x_prior_1
        self.x_post_2[0, :] += self.x_prior_2


# TODO
class SDGPR_H(AEP_Model):

    def __init__(self, x_train, y_train, no_pseudos, hidden_sizes, lik='Gaussian'):
        super(SDGPR, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = x_train.shape[1]
        self.size = [Din] + hidden_sizes + [Dout]
        self.L = L = len(self.size) - 1
        if not isinstance(no_pseudos, (list, tuple)):
            self.Ms = Ms = [no_pseudos for i in range(L)]
        else:
            self.Ms = Ms = no_pseudos
        self.x_train = x_train

        self.sgp_layers = []
        for i in range(L):
            Din_i = self.size[i]
            Dout_i = self.size[i+1]
            M_i = self.Ms[i]
            self.sgp_layers.append(SGP_Layer(N, Din_i, Dout_i, M_i))

        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

    def objective_function(self, params, mb_size, alpha=1.0):
        N = self.N
        if mb_size >= N:
            xb = self.x_train
            yb = self.y_train
        else:
            idxs = np.random.randint(0, N, mb_size)
            xb = self.x_train[idxs, :]
            yb = self.y_train[idxs, :]
        batch_size = yb.shape[0]
        scale_logZ = - N * 1.0 / batch_size / alpha
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        
        # update model with new hypers
        self.update_hypers(params)
        for layer in self.sgp_layers:
            layer.compute_cavity(alpha)

        #### propagate x cavity forward
        mout, vout, psi1, psi2 = [], [], [], []
        for i in range(0, self.L):
            layer = self.sgp_layers[i]
            if i == 0:
                # first layer
                m0, v0, kfu0 = layer.forward_prop_thru_cav(xb)
                mout.append(m0)
                vout.append(v0)
                psi1.append(kfu0)
                psi2.append(None)
            else:
                mi, vi, psi1i, psi2i = layer.forward_prop_thru_cav(
                    mout[i-1], vout[i-1])
                mout.append(mi)
                vout.append(vi)
                psi1.append(psi1i)
                psi2.append(psi2i)

        # compute logZ and gradients
        logZ, dm, dv = self.lik_layer.compute_log_Z(
            mout[-1], vout[-1], yb, alpha)
        logZ_scale = scale_logZ * logZ
        dmi = scale_logZ * dm
        dvi = scale_logZ * dv
        grad_list = []
        for i in range(self.L-1, -1, -1):
            layer = self.sgp_layers[i]
            if i == 0:
                grad_hyper = layer.backprop_grads_reg(
                    mout[i], vout[i], dmi, dvi, psi1[i], xb, alpha)
            else:
                grad_hyper, grad_input = layer.backprop_grads_lvm(
                    mout[i], vout[i], dmi, dvi, psi1[i], psi2[i], 
                    mout[i-1], vout[i-1], alpha)
                dmi, dvi = grad_input['mx'], grad_input['vx']
            grad_list.insert(0, grad_hyper)

        lik_grad_hyper = self.lik_layer.backprop_grads(
            mout[-1], vout[-1], dm, dv, alpha, scale_logZ)
        
        grad_all = {}
        for i, grad in enumerate(grad_list):
            for key in grad.keys():
                grad_all[key + '_%d'%i] = grad[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute objective
        sgp_contrib = 0
        for layer in self.sgp_layers:
            sgp_contrib += layer.compute_phi(alpha)
        energy = logZ_scale + sgp_contrib

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def predict_f(self, inputs):
        if not self.updated:
            for layer in self.sgp_layers:
                layer.update_posterior_for_prediction()
            self.updated = True
        for i, layer in enumerate(self.sgp_layers):
            if i == 0:
                mf, vf = layer.forward_prop_thru_post(inputs)
            else:
                mf, vf = layer.forward_prop_thru_post(mf, vf)
        return mf, vf

    def sample_f(self, inputs, no_samples=1):
        if not self.updated:
            for layer in self.sgp_layers:
                layer.update_posterior_for_prediction()
            self.updated = True
        K = no_samples
        fs = np.zeros((inputs.shape[0], self.Dout, K))
        # TODO: remove for loop here
        for k in range(K):
            inputs_k = inputs
            for layer in self.sgp_layers:
                outputs = layer.sample(inputs_k)
                inputs_k = outputs
            fs[:, :, k] = outputs
        return fs

    def predict_y(self, inputs):
        mf, vf = self.predict_f(inputs)
        my, vy = self.lik_layer.forward_prop_thru_post(mf, vf)
        return my, vy

    def init_hypers(self, y_train):
        init_params = dict()
        for i in range(self.L):
            if i == 0:
                sgp_params = self.sgp_layers[i].init_hypers(
                    self.x_train, 
                    key_suffix='_%d'%i)
            else:
                sgp_params = self.sgp_layers[i].init_hypers(
                    key_suffix='_%d'%i)
            init_params.update(sgp_params)

        lik_params = self.lik_layer.init_hypers()
        init_params.update(lik_params)
        return init_params

    def get_hypers(self):
        params = dict()
        for i in range(self.L):
            sgp_params = self.sgp_layers[i].get_hypers(key_suffix='_%d'%i)
            params.update(sgp_params)
        lik_params = self.lik_layer.get_hypers()
        params.update(lik_params)
        return params

    def update_hypers(self, params):
        for i, layer in enumerate(self.sgp_layers):
            layer.update_hypers(params, key_suffix='_%d'%i)
        self.lik_layer.update_hypers(params)
