import pprint
pp = pprint.PrettyPrinter(indent=4)
import sys
import math
import numpy as np
import scipy.linalg as npalg
import scipy.stats as stats
from fitc_layer import FITC_Layer
from eq_kernel import *
import copy
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import os
from tools import *
import pdb
from scipy.optimize import minimize


class GP_AEP_Layer:

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


class GPLVM_AEP:

    def __init__(self, y_train, hidden_size, no_pseudo,
                 lik='Gaussian'):

        self.lik = lik
        self.y_train = y_train
        self.N_train, self.output_size = y_train.shape
        N_train = self.N_train
        self.hidden_size = hidden_size
        self.no_pseudo = no_pseudo

        # create a layer
        self.sgp_layer = GP_AEP_Layer(self.N_train, self.hidden_size,
                                     self.output_size, self.no_pseudo,
                                     lik=lik)

        # natural params for latent variables
        self.factor_x1 = np.zeros((N_train, hidden_size))
        self.factor_x2 = np.zeros((N_train, hidden_size))

        self.prior_x1 = 0
        self.prior_x2 = 1

        self.x_post_1 = np.zeros((N_train, hidden_size))
        self.x_post_2 = np.zeros((N_train, hidden_size))

        self.fixed_params = []

    # @profile
    def train_overhead(self, params, alpha=1.0):
        toprint = False
        # update layer with new hypers
        t1 = time.time()
        self.fitc_layer.update_hypers(params)
        self.factor_x1 = params['x1']
        self.factor_x2 = np.exp(2 * params['x2'])
        # self.factor_x2 = sigmoid(params['x2'])
        t2 = time.time()
        if toprint:
            print "update hypers %.4fs" % (t2 - t1)

        # update Kuu given new hypers
        t1 = time.time()
        self.fitc_layer.compute_kuu()
        t2 = time.time()
        if toprint:
            print "compute Kuu %.4fs" % (t2 - t1)

        # compute mu and Su for each layer
        t1 = time.time()
        self.fitc_layer.update_posterior()
        t2 = time.time()
        if toprint:
            print "compute posterior %.4fs" % (t2 - t1)

        # compute muhat and Suhat for each layer
        t1 = time.time()
        self.fitc_layer.compute_cavity(alpha=alpha)
        t2 = time.time()
        if toprint:
            print "compute cavity %.4fs" % (t2 - t1)

    # @profile
    def objective_function(self, params, idxs, yb, N_train, alpha=1.0):
        toprint = False
        N_batch = yb.shape[0]
        layer = self.fitc_layer
        scale_logZ = - N_train * 1.0 / N_batch / alpha

        beta = (N_train - alpha) * 1.0 / N_train
        scale_poste = N_train * 1.0 / alpha - 1.0
        scale_cav = - N_train * 1.0 / alpha
        scale_prior = 1

        self.train_overhead(params, alpha=alpha)

        # reset gradient placeholders
        grad_all = {}

        M = layer.no_pseudos
        Din = layer.hidden_size
        Dout = layer.output_size

        t1 = time.time()
        # compute cavity
        cav_1 = self.prior_x1 + (1.0 - alpha) * self.factor_x1[idxs, :]
        cav_2 = self.prior_x2 + (1.0 - alpha) * self.factor_x2[idxs, :]
        vx = 1.0 / cav_2
        mx = cav_1 / cav_2
        logZi, grad_logZ = layer.compute_logZ_and_gradients(mx, vx, yb, alpha)
        # print grad_logZ['sf']
        t2 = time.time()
        if toprint:
            print "compute logZ and grads %.4fs" % (t2 - t1)

        time1 = time.time()

        grad_all['sf'] = scale_logZ * grad_logZ['sf']
        grad_all['ls'] = scale_logZ * grad_logZ['ls']
        if self.lik.lower() == 'gaussian':
            grad_all['sn'] = scale_logZ * grad_logZ['sn']
        grad_all['zu'] = scale_logZ * grad_logZ['zu']

        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        Minner = 0

        mu = layer.mu
        Su = layer.Su
        Spmm = layer.Splusmm
        muhat = layer.muhat
        Suhat = layer.Suhat
        Spmmhat = layer.Splusmmhat
        Kuuinv = layer.Kuuinv
        Kuu = layer.Kuu
        theta2 = layer.theta_2
        theta1_R = layer.theta_1_R

        dlogZ_dAhat = grad_logZ['Ahat']
        dlogZ_dBhat = grad_logZ['Bhat']
        dlogZ_dvcav = np.einsum('ab,dbc,ce->dae', Kuuinv, dlogZ_dBhat, Kuuinv)
        dlogZ_dmcav = 2 * np.einsum('dab,db->da', dlogZ_dvcav, muhat) \
            + np.einsum('ab,db->da', Kuuinv, dlogZ_dAhat)

        dlogZ_dvcav_via_mcav = beta * \
            np.einsum('da,db->dab', dlogZ_dmcav, theta2)
        dlogZ_dvcav += dlogZ_dvcav_via_mcav
        dlogZ_dvcavinv = - \
            np.einsum('dab,dbc,dce->dae', Suhat, dlogZ_dvcav, Suhat)
        dlogZ_dtheta1 = beta * dlogZ_dvcavinv
        dlogZ_dtheta2 = beta * np.einsum('dab,db->da', Suhat, dlogZ_dmcav)
        dlogZ_dKuuinv_via_vcav = np.sum(dlogZ_dvcavinv, axis=0)

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dlogZ_dKuuinv_via_Ahat = np.einsum('da,db->ab', dlogZ_dAhat, muhat)
        KuuinvSmmd = np.einsum('ab,dbc->dac', Kuuinv, Spmmhat)
        dlogZ_dKuuinv_via_Bhat = 2 * np.einsum('dab,dac->bc', KuuinvSmmd, dlogZ_dBhat) \
            - np.sum(dlogZ_dBhat, axis=0)
        dlogZ_dKuuinv = dlogZ_dKuuinv_via_Ahat + \
            dlogZ_dKuuinv_via_Bhat + dlogZ_dKuuinv_via_vcav
        Minner = scale_poste * np.sum(Spmm, axis=0) + scale_cav * \
            np.sum(Spmmhat, axis=0) - 2.0 * scale_logZ * dlogZ_dKuuinv

        dtheta1 = scale_poste * -0.5 * Spmm + scale_cav * \
            beta * -0.5 * Spmmhat + scale_logZ * dlogZ_dtheta1
        dtheta2 = scale_poste * mu + scale_cav * \
            beta * muhat + scale_logZ * dlogZ_dtheta2
        dtheta1_T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum('dab,dbc->dac', theta1_R, dtheta1 + dtheta1_T)

        grad_all['eta1_R'] = np.zeros([Dout, M * (M + 1) / 2])
        grad_all['eta2'] = dtheta2
        for d in range(Dout):
            dtheta1_R_d = dtheta1_R[d, :, :]
            theta1_R_d = theta1_R[d, :, :]
            dtheta1_R_d[diag_ind] = dtheta1_R_d[
                diag_ind] * theta1_R_d[diag_ind]
            dtheta1_R_d = dtheta1_R_d[triu_ind]
            grad_all['eta1_R'][d, :] = dtheta1_R_d.reshape(
                (dtheta1_R_d.shape[0], ))

        M_all = 0.5 * (scale_prior * Dout * Kuuinv +
                       np.dot(Kuuinv, np.dot(Minner, Kuuinv)))
        dhyp = d_trace_MKzz_dhypers(
            2 * layer.ls, 2 * layer.sf, layer.zu, M_all, Kuu)

        grad_all['sf'] += 2 * dhyp[0]
        grad_all['ls'] += 2 * dhyp[1]
        grad_all['zu'] += dhyp[2]

        # compute grad wrt x params
        scale_x = N_train * 1.0 / N_batch
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]

        t1 = t01 + (1.0 - alpha) * t11
        t2 = t02 + (1.0 - alpha) * t12

        dcav_dt11 = (1.0 - alpha) * t1 / t2
        dcav_dt12 = (1.0 - alpha) * (- 0.5 * t1**2 / t2**2 - 0.5 / t2)

        dlogZ_dt11 = (1.0 - alpha) * grad_logZ['mx'] / t2
        dlogZ_dt12 = (1.0 - alpha) * \
            (- grad_logZ['mx'] * t1 / t2**2 - grad_logZ['vx'] / t2**2)

        t1 = t01 + t11
        t2 = t02 + t12

        dpost_dt11 = t1 / t2
        dpost_dt12 = - 0.5 * t1**2 / t2**2 - 0.5 / t2

        grad_all['x1'] = (scale_logZ * dlogZ_dt11
                          + scale_x * (- 1.0 / alpha * dcav_dt11 - (1.0 - 1.0 / alpha) * dpost_dt11))
        grad_all['x2'] = (scale_logZ * dlogZ_dt12
                          + scale_x * (- 1.0 / alpha * dcav_dt12 - (1.0 - 1.0 / alpha) * dpost_dt12))
        grad_all['x2'] *= 2 * t12

        if toprint:
            print "merge gradients %.4fs" % (time.time() - time1)

        time1 = time.time()

        phi_prior = layer.compute_phi_prior()
        phi_poste = layer.compute_phi_posterior()
        phi_cavity = layer.compute_phi_cavity()

        phi_prior_x = self.compute_phi_prior_x(idxs)
        phi_poste_x = self.compute_phi_posterior_x(idxs)
        phi_cavity_x = self.compute_phi_cavity_x(idxs, alpha)

        energy = 0
        energy = scale_prior * phi_prior + scale_poste * \
            phi_poste + scale_cav * phi_cavity + scale_logZ * logZi
        energy += scale_x * (phi_prior_x - 1.0 / alpha *
                             phi_cavity_x - (1.0 - 1.0 / alpha) * phi_poste_x)

        if toprint:
            print "compute energy %.4fs" % (time.time() - time1)

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def compute_phi_prior_x(self, idxs):
        t1 = self.prior_x1
        t2 = self.prior_x2
        m = t1 / t2
        v = 1.0 / t2
        Nb = idxs.shape[0]
        hidden_size = self.hidden_size
        return 0.5 * Nb * hidden_size * (m**2 / v + np.log(v))

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

    def predict_given_inputs(self, inputs, add_noise=False):
        my, vy = self.fitc_layer.output_probabilistic(inputs, add_noise=False)
        return my, vy

    def get_posterior_x(self):
        post_1 = self.prior_x1 + self.factor_x1
        post_2 = self.prior_x2 + self.factor_x2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def impute_missing(self, y, missing_mask, alpha=0.5, no_iters=10, add_noise=False):
        # find latent conditioned on observed variables
        N_test = y.shape[0]
        Q = self.hidden_size
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
                self.fitc_layer.compute_logZ_and_gradients_imputation(
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
        # pdb.set_trace()

        # propagate x forward to predict missing points
        my, vy = self.fitc_layer.forward_propagation_thru_post(
            post_m, post_v, add_noise=add_noise)

        return my, vy

    def initialise_hypers(self, y_train):
        Din = self.hidden_size
        init_params = self.fitc_layer.init_hypers()
        # if self.lik.lower() == 'gaussian':
        #     post_m = PCA_reduce(y_train, Din)
        # else:
        #     post_m = np.random.randn(y_train.shape[0], Din)
        post_m = PCA_reduce(y_train, Din)
        post_m_mean = np.mean(post_m, axis=0)
        post_m_std = np.std(post_m, axis=0)
        post_m = (post_m - post_m_mean) / post_m_std
        post_v = 0.1 * np.ones_like(post_m)
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        init_params['x1'] = post_1
        init_params['x2'] = np.log(post_2 - 1) / 2
        # init_params['x2'] = inverse_sigmoid(post_2 - 1)
        return init_params

    def get_hypers(self):
        params = self.fitc_layer.get_hypers()
        params['x1'] = self.factor_x1
        params['x2'] = np.log(self.factor_x2) / 2.0
        return params

    def set_fixed_params(self, params):
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)

    def optimise(self, method='L-BFGS-B', tol=None,
                 reinit_hypers=True, callback=None, maxiter=1000, alpha=0.5,
                 adam_lr=0.05, **kargs):
        if reinit_hypers:
            init_params_dict = self.initialise_hypers(self.y_train)
        else:
            init_params_dict = self.get_hypers()

        init_params_vec, params_args = flatten_dict(init_params_dict)

        N_train = self.N_train
        idxs = np.arange(N_train)
        yb = self.y_train

        try:
            if method.lower() == 'adam':
                results = adam(objective_wrapper, init_params_vec,
                               step_size=adam_lr,
                               maxiter=maxiter,
                               args=(params_args, self, idxs, yb, N_train, alpha))
            else:
                options = {'maxiter': maxiter, 'disp': True, 'gtol': 1e-8}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, idxs, yb, N_train, alpha),
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
