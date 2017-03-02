import sys
import os
import numpy as np
from eq_kernel import *
import scipy.linalg as spalg
import scipy as scipy
import pprint as pp
import time
from tools import *
import pdb

from scipy import special

from scipy.cluster.vq import vq, kmeans2

from scipy.spatial.distance import cdist

import matplotlib.pylab as plt


class FITC_Layer:

    def __init__(self, Ntrain, hidden_size, output_size, no_pseudo, lik):
        self.lik = lik
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.no_pseudos = no_pseudo
        self.Ntrain = Ntrain
        self.jitter = 1e-4
        self.no_output_noise = self.lik.lower() != 'gaussian'

        self.ones_M = np.ones(no_pseudo)
        self.ones_D = np.ones(hidden_size)
        self.ones_M_ls = 0

        Din = self.hidden_size
        Dout = self.output_size
        M = self.no_pseudos

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
        if not self.no_output_noise:
            self.sn = 0

        # and natural parameters
        self.theta_1_R = np.zeros([Dout, M, M])
        self.theta_2 = np.zeros([Dout, M, ])
        self.theta_1 = np.zeros([Dout, M, M])

        # terms that are common to all datapoints in each minibatch
        self.Ahat = np.zeros([Dout, M, ])
        self.Bhat = np.zeros([Dout, M, M])
        self.A = np.zeros([Dout, M, ])
        self.B_det = np.zeros([Dout, M, M])
        self.B_sto = np.zeros([Dout, M, M])

    def compute_phi_prior(self):
        logZ_prior = 0
        Dout = self.output_size
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        logZ_prior += Dout * 0.5 * logdet

        return logZ_prior

    def compute_phi_posterior(self):
        logZ_posterior = 0
        mu = self.mu
        Su = self.Su
        (sign, logdet) = np.linalg.slogdet(Su)
        phi_posterior = 0.5 * np.sum(logdet)
        phi_posterior += 0.5 * \
            np.sum(mu * np.linalg.solve(Su, mu))
        return phi_posterior


    def compute_phi_cavity(self):
        logZ_posterior = 0
        mu = self.muhat
        Su = self.Suhat
        (sign, logdet) = np.linalg.slogdet(Su)
        phi_cavity = 0.5 * np.sum(logdet)
        phi_cavity += 0.5 * \
            np.sum(mu * np.linalg.solve(Su, mu))
        return phi_cavity


    def output_probabilistic(self, x, add_noise=False):
        Din = self.hidden_size
        Dout = self.output_size
        M = self.no_pseudos
        ls = self.ls
        sf = self.sf
        sf2 = np.exp(2.0 * sf)
        psi1 = compute_kernel(2 * ls, 2 * sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bhatpsi2 = np.einsum('dab,na,nb->nd', self.B_det, psi1, psi1)
        vout = sf2 + Bhatpsi2
        if add_noise and self.lik.lower() == 'gaussian':
            sn2 = np.exp(2.0 * self.sn)
            vout += sn2

        return mout, vout

    def forward_propagation_thru_post(self, mx, vx, add_noise=True):
        ls = self.ls
        sf = self.sf
        sf2 = np.exp(2.0 * sf)
        N = mx.shape[0]
        psi0 = sf2
        zu = self.zu

        psi1, psi2 = compute_psi_weave(2 * ls, 2 * sf, mx, vx, zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.B_sto, psi2)
        vout = psi0 + Bhatpsi2 - mout**2

        if add_noise and self.lik.lower() == 'gaussian':
            sn2 = np.exp(2.0 * self.sn)
            vout += sn2

        return mout, vout


    def compute_logZ_and_gradients_imputation(
            self, mx, vx, y, missing_mask, 
            alpha=1.0, add_noise=False, gh_deg=10):
        Din = self.hidden_size
        Dout = self.output_size
        M = self.no_pseudos
        ls = self.ls
        sf = self.sf
        sf2 = np.exp(2.0 * sf)
        N = mx.shape[0]
        psi0 = sf2
        zu = self.zu

        psi1, psi2 = compute_psi_weave(2 * ls, 2 * sf, mx, vx, zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.B_sto, psi2)
        vout = psi0 + Bhatpsi2 - mout**2

        # compute logZ
        if self.lik == 'Gaussian':
            # real valued data, gaussian lik
            sn2 = np.exp(2.0 * self.sn)
            vout += sn2 / alpha
            logZ = np.sum(-0.5 * (np.log(2 * np.pi * vout) +
                                  (y - mout)**2 / vout))
            logZ += N * Dout * (0.5 * np.log(2 * np.pi * sn2 / alpha)
                                - 0.5 * alpha * np.log(2 * np.pi * sn2))
            dlogZ_dm = (y - mout) / vout
            dlogZ_dv = -0.5 / vout + 0.5 * (y - mout)**2 / vout**2
        elif self.lik == 'Probit':
            # binary data probit likelihood
            if alpha == 1.0:
                t = y * mout / np.sqrt(1 + vout)
                Z = 0.5 * (1 + special.erf(t / np.sqrt(2)))
                eps = 1e-16
                logZ = np.sum(np.log(Z + eps) * missing_mask)

                dlogZ_dt = 1 / (Z + eps) * 1 / np.sqrt(2 *
                                                       np.pi) * np.exp(-t**2.0 / 2)
                dlogZ_dt *= missing_mask
                dt_dm = y / np.sqrt(1 + vout)
                dt_dv = -0.5 * y * mout / (1 + vout)**1.5
                dlogZ_dm = dlogZ_dt * dt_dm
                dlogZ_dv = dlogZ_dt * dt_dv
            else:
                gh_x, gh_w = self._gh_points(gh_deg)
                gh_x = gh_x[:, np.newaxis, np.newaxis]
                gh_w = gh_w[:, np.newaxis, np.newaxis]
                ts = gh_x * np.sqrt(2*vout[np.newaxis, :, :]) + mout[np.newaxis, :, :]
                eps = 1e-8
                pdfs = 0.5 * (1 + special.erf(y*ts / np.sqrt(2))) + eps
                Ztilted = np.sum(pdfs**alpha * gh_w, axis=0) / np.sqrt(np.pi)
                logZ = np.sum(np.log(Ztilted) * missing_mask)
                
                a = pdfs**(alpha-1.0)*np.exp(-ts**2/2)
                dZdm = np.sum(gh_w * a, axis=0) * y * alpha / np.pi / np.sqrt(2)
                dlogZ_dm = (dZdm / Ztilted + eps) * missing_mask

                dZdv = np.sum(gh_w * (a*gh_x), axis=0) * y * alpha / np.pi / np.sqrt(2) / np.sqrt(2*vout)
                dlogZ_dv = (dZdv / Ztilted + eps) * missing_mask


        # compute grads wrt Ahat and Bhat
        dlogZ_dm_all = dlogZ_dm - 2 * dlogZ_dv * mout
        # compute grads wrt psi1 and psi2
        dpsi1 = np.einsum('nd,dm->nm', dlogZ_dm_all, self.A)
        dpsi2 = np.einsum('nd,dab->nab', dlogZ_dv, self.B_sto)
        dsf2, dls, dzu, dmx, dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, np.exp(ls), sf2, mx, vx, zu)

        return logZ, mout, vout, dmx, dvx

    __gh_points = None
    def _gh_points(self, T=20):
        if self.__gh_points is None:
            self.__gh_points = np.polynomial.hermite.hermgauss(T)
        return self.__gh_points

    # @profile
    def compute_logZ_and_gradients(self, mx, vx, y, alpha=1.0, gh_deg=10):
        Din = self.hidden_size
        Dout = self.output_size
        M = self.no_pseudos
        ls = self.ls
        sf = self.sf
        sf2 = np.exp(2.0 * sf)
        N = mx.shape[0]
        psi0 = sf2
        zu = self.zu

        psi1, psi2 = compute_psi_weave(2 * ls, 2 * sf, mx, vx, zu)
        mout = np.einsum('nm,dm->nd', psi1, self.Ahat)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.Bhat, psi2)
        vout = psi0 + Bhatpsi2 - mout**2

        # compute logZ
        if self.lik == 'Gaussian':
            # real valued data, gaussian lik
            sn2 = np.exp(2.0 * self.sn)
            vout += sn2 / alpha
            logZ = np.sum(-0.5 * (np.log(2 * np.pi * vout) +
                                  (y - mout)**2 / vout))
            logZ += N * Dout * (0.5 * np.log(2 * np.pi * sn2 / alpha)
                                - 0.5 * alpha * np.log(2 * np.pi * sn2))
            dlogZ_dm = (y - mout) / vout
            dlogZ_dv = -0.5 / vout + 0.5 * (y - mout)**2 / vout**2
        elif self.lik == 'Probit':
            # binary data probit likelihood
            if alpha == 1:
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
                gh_x, gh_w = self._gh_points(gh_deg)
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

        # compute grads wrt Ahat and Bhat
        dlogZ_dm_all = dlogZ_dm - 2 * dlogZ_dv * mout
        dAhat = np.einsum('nd,nm->dm', dlogZ_dm_all, psi1)
        dBhat = np.einsum('nd,nab->dab', dlogZ_dv, psi2)
        # compute grads wrt psi1 and psi2
        dpsi1 = np.einsum('nd,dm->nm', dlogZ_dm_all, self.Ahat)
        dpsi2 = np.einsum('nd,dab->nab', dlogZ_dv, self.Bhat)
        dsf2, dls, dzu, dmx, dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, np.exp(ls), sf2, mx, vx, zu)

        sum_dlogZ_dv = np.sum(dlogZ_dv)
        dls *= np.exp(ls)
        dsf2 += sum_dlogZ_dv
        dsf = 2 * sf2 * dsf2

        if self.lik == 'Gaussian':
            dsn = sum_dlogZ_dv * 2 * sn2 / alpha + N * Dout * (1 - alpha)

        grads = {'ls': dls, 'sf': dsf, 'zu': dzu,
                 'Ahat': dAhat, 'Bhat': dBhat, 'mx': dmx, 'vx': dvx}

        if self.lik.lower() == 'gaussian':
            grads['sn'] = dsn

        return logZ, grads

    def compute_kuu(self):
        # update kuu and kuuinv
        ls = self.ls
        sf = self.sf
        Dout = self.output_size
        M = self.no_pseudos
        zu = self.zu
        self.Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        self.Kuu += np.diag(self.jitter * np.ones((M, )))
        # self.Kuuinv = matrixInverse(self.Kuu)
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def compute_cavity(self, alpha=1.0):
        # compute the leave one out moments
        beta = (self.Ntrain - alpha) * 1.0 / self.Ntrain

        Dout = self.output_size
        Kuuinv = self.Kuuinv
        for d in range(Dout):
            self.Suhatinv[d, :, :] = Kuuinv + beta * self.theta_1[d, :, :]
            ShatinvMhat = beta * self.theta_2[d, :]
            # Shat = matrixInverse(self.Suhatinv[d, :, :])
            Shat = np.linalg.inv(self.Suhatinv[d, :, :])
            self.Suhat[d, :, :] = Shat
            mhat = np.dot(Shat, ShatinvMhat)
            self.muhat[d, :] = mhat
            self.Ahat[d, :] = np.dot(Kuuinv, mhat)
            Smm = Shat + np.outer(mhat, mhat)
            self.Splusmmhat[d, :, :] = Smm
            self.Bhat[d, :, :] = np.dot(Kuuinv, np.dot(Smm, Kuuinv)) - Kuuinv

    def update_posterior(self):
        # compute the posterior approximation
        Dout = self.output_size
        Kuuinv = self.Kuuinv
        for d in range(Dout):
            Sinv = Kuuinv + self.theta_1[d, :, :]
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
        Dout = self.output_size
        for d in range(Dout):
            Kuuinvd = self.Kuuinv
            Sinv = Kuuinvd + self.theta_1[d, :, :]
            SinvM = self.theta_2[d, :]
            S = matrixInverse(Sinv)
            self.Su[d, :, :] = S
            m = np.dot(S, SinvM)
            self.mu[d, :] = m

            self.A[d, :] = np.dot(Kuuinvd, m)
            Smm = S + np.outer(m, m)
            self.Splusmm[d, :, :] = Smm
            self.B_det[d, :, :] = np.dot(Kuuinvd, np.dot(S, Kuuinvd)) - Kuuinvd
            self.B_sto[d, :, :] = np.dot(Kuuinvd, np.dot(Smm, Kuuinvd)) - Kuuinvd

    def init_hypers(self):
        # dict to hold hypers, inducing points and parameters of q(U)
        params = {'ls': [],
                  'sf': [],
                  'zu': [],
                  'eta1_R': [],
                  'eta2': [],
                  'x1': [],
                  'x2': []}

        Ntrain = self.Ntrain
        M = self.no_pseudos
        Din = self.hidden_size
        Dout = self.output_size

        # ls = np.log(np.random.rand(Din, ))
        ls = np.log(np.ones((Din, )) + 0.1 * np.random.rand(Din, ))
        sf = np.log(np.array([0.5]))

        params['sf'] = sf
        params['ls'] = ls
        if not self.no_output_noise:
            sn = np.log(np.array([0.0001]))
            params['sn'] = sn

        eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        eta2 = np.zeros((Dout, M))
        zu = np.tile(np.linspace(-1.2, 1.2, M).reshape((M, 1)), (1, Din))
        zu += 0.1 * np.random.randn(zu.shape[0], zu.shape[1])
        # zu = np.random.randn(M, Din)
        Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        Kuu += np.diag(self.jitter * np.ones((M, )))
        Kuuinv = matrixInverse(Kuu)

        for d in range(Dout):
            mu = np.linspace(-1.2, 1.2, M).reshape((M, 1))
            mu += 0.1 * np.random.randn(M, 1)
            alpha = 0.1 * np.random.rand(M)
            Su = np.diag(alpha)
            Suinv = np.diag(1 / alpha)

            theta2 = np.dot(Suinv, mu)
            theta1 = Suinv
            R = np.linalg.cholesky(theta1).T

            triu_ind = np.triu_indices(M)
            diag_ind = np.diag_indices(M)
            R[diag_ind] = np.log(R[diag_ind])
            eta1_d = R[triu_ind].reshape((M * (M + 1) / 2,))
            eta2_d = theta2.reshape((M,))

            eta1_R[d, :] = eta1_d
            eta2[d, :] = eta2_d

        params['zu'] = zu
        params['eta1_R'] = eta1_R
        params['eta2'] = eta2

        return params

    def get_hypers(self):
        params = {}
        M = self.no_pseudos
        Din = self.hidden_size
        Dout = self.output_size
        params['ls'] = self.ls
        params['sf'] = self.sf
        if not self.no_output_noise:
            params['sn'] = self.sn
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        params_eta2 = self.theta_2
        params_eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        params_zu_i = self.zu

        for d in range(Dout):
            Rd = self.theta_1_R[d, :, :]
            Rd[diag_ind] = np.log(Rd[diag_ind])
            params_eta1_R[d, :] = Rd[triu_ind]

        params['zu'] = self.zu
        params['eta1_R'] = params_eta1_R
        params['eta2'] = params_eta2
        return params

    def update_hypers(self, params):
        M = self.no_pseudos
        Din = self.hidden_size
        Dout = self.output_size
        self.ls = params['ls']
        self.ones_M_ls = np.outer(self.ones_M, 1.0 / np.exp(2 * self.ls))
        self.sf = params['sf']
        if not (self.no_output_noise):
            self.sn = params['sn']

        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        zu = params['zu']
        self.zu = zu

        for d in range(Dout):
            theta_m_d = params['eta2'][d, :]
            theta_R_d = params['eta1_R'][d, :]
            R = np.zeros((M, M))
            R[triu_ind] = theta_R_d.reshape(theta_R_d.shape[0], )
            R[diag_ind] = np.exp(R[diag_ind])
            self.theta_1_R[d, :, :] = R
            self.theta_1[d, :, :] = np.dot(R.T, R)
            self.theta_2[d, :] = theta_m_d
