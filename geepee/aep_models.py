"""Sparse approximations for Gaussian process models

Models implemented include GP Gaussian regression/Probit classification,
GP latent variable model, GP state space model and Deep GPs

Inference and learning using approximate EP (or Black-box alpha)

"""
import sys
import math
import numpy as np
import scipy.linalg as npalg
import matplotlib.pyplot as plt
import time
import pdb
import pprint
import collections

from config import *
from utils import profile
from kernels import *
from base_models import Base_Model
from base_models import Base_SGP_Layer, Base_SGPR, Base_SGPLVM, Base_SDGPR, Base_SGPSSM


class SGP_Layer(Base_SGP_Layer):
    """Summary
    
    Attributes:
        Ahat (TYPE): Description
        Bhat_det (TYPE): Description
        Bhat_sto (TYPE): Description
        muhat (TYPE): Description
        Splusmmhat (TYPE): Description
        Suhat (TYPE): Description
        Suhatinv (TYPE): Description
    """
    def __init__(self, no_train, input_size, output_size, no_pseudo, 
        nat_param=True):
        """Initialisation
        
        Args:
            no_train (int): Number of training points
            input_size (int): Number of input dimensions
            output_size (int): Number of output dimensions
            no_pseudo (int): Number of pseudo-points
        """
        super(SGP_Layer, self).__init__(
            no_train, input_size, output_size, no_pseudo, nat_param)
        # variables for the cavity distribution
        Dout, M = self.Dout, self.M
        self.muhat = np.zeros([Dout, M, ])
        self.Suhat = np.zeros([Dout, M, M])
        self.Suhatinv = np.zeros([Dout, M, M])
        self.Splusmmhat = np.zeros([Dout, M, M])

        # terms that are common to all datapoints in each minibatch
        self.Ahat = np.zeros([Dout, M, ])
        self.Bhat_sto = np.zeros([Dout, M, M])
        self.Bhat_det = np.zeros([Dout, M, M])

    def compute_phi_prior(self):
        """Compute the log-partition of the prior p(u)
        
        Returns:
            float: log-partition
        """
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        logZ_prior = self.Dout * 0.5 * logdet
        return logZ_prior

    def compute_phi_posterior(self):
        """Compute the log-partition of the posterior q(u)
        
        Returns:
            float: log-partition
        """
        (sign, logdet) = np.linalg.slogdet(self.Su)
        phi_posterior = 0.5 * np.sum(logdet)
        phi_posterior += 0.5 * np.sum(self.mu * np.linalg.solve(
            self.Su, self.mu))
        return phi_posterior

    def compute_phi_cavity(self):
        """Compute the log-partition of the cavity distribution
        
        Returns:
            float: log-partition
        """
        logZ_posterior = 0
        (sign, logdet) = np.linalg.slogdet(self.Suhat)
        phi_cavity = 0.5 * np.sum(logdet)
        phi_cavity += 0.5 * np.sum(self.muhat * np.linalg.solve(
            self.Suhat, self.muhat))
        return phi_cavity

    def compute_phi(self, alpha=1.0):
        """Compute the weighted sum of the log-partitions of prior, post and cav  
        
        Args:
            alpha (float, optional): power parameter
        
        Returns:
            float: weighted sum of the log-partitions in the PEP energy
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

    @profile
    def backprop_grads_lvm_mm(self, m, v, dm, dv, psi1, psi2, mx, vx, alpha=1.0):
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
            'eta1_R': deta1_R, 'eta2': deta2}
        grad_input = {'mx': dmx, 'vx': dvx}

        return grad_hyper, grad_input

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

    @profile
    def backprop_grads_reg(self, m, v, dm, dv, kfu, x, alpha=1.0):
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
        Suinv = self.Suinv
        Spmm = self.Splusmm
        muhat = self.muhat
        Suhat = self.Suhat
        Suhatinv = self.Suhatinv
        Spmmhat = self.Splusmmhat
        Kuuinv = self.Kuuinv
        Kuu = self.Kuu
        kfuKuuinv = np.dot(kfu, Kuuinv)

        beta = (N - alpha) * 1.0 / N
        scale_post = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1

        # compute grads wrt kfu
        dkfu_m = np.einsum('nd,dm->nm', dm, self.Ahat)
        dkfu_v = 2 * np.einsum('nd,dab,na->nb', dv, self.Bhat_det, kfu)
        dkfu = dkfu_m + dkfu_v
        dsf2, dls, dzu = compute_kfu_derivatives(
            dkfu, kfu, ls, sf2, x, self.zu)
        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        # compute grads wrt cavity mean and covariance
        dmucav = np.einsum('nd,nm->dm', dm, kfuKuuinv)
        dSucav = np.einsum('na,nd,nb->dab', kfuKuuinv, dv, kfuKuuinv)
        # add in contribution from the normalising factor
        Suinvmuhat = np.einsum('dab,db->da', Suhatinv, muhat)
        dmucav += scale_cav * Suinvmuhat
        dSucav += scale_cav * (
            0.5 * Suhatinv 
            - 0.5 * np.einsum('da,db->dab', Suinvmuhat, Suinvmuhat))
        deta1_R_cav, deta2_cav, dKuuinv_via_cav = \
            self.compute_cav_grad_u(dmucav, dSucav, alpha)

        # compute grads wrt posterior mean and covariance
        Suinvmu = np.einsum('dab,db->da', Suinv, mu)
        dmu = scale_post * Suinvmu
        dSu = scale_post * (
            0.5 * Suinv - 0.5 * np.einsum('da,db->dab', Suinvmu, Suinvmu))
        deta1_R_post, deta2_post, dKuuinv_via_post = \
            self.compute_posterior_grad_u(dmu, dSu)

        # contrib from phi prior term
        dKuuinv_via_prior = - 0.5 * self.Dout * Kuu

        deta1_R = deta1_R_cav + deta1_R_post
        deta2 = deta2_cav + deta2_post
        dKuuinv_via_phi = dKuuinv_via_cav + dKuuinv_via_post + dKuuinv_via_prior

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dAhat = np.einsum('nd,nm->dm', dm, kfu)
        dKuuinv_m = np.einsum('da,db->ab', dAhat, muhat)
        KuuinvSmmd = np.einsum('ab,dbc->dac', Kuuinv, Suhat)
        dBhat = np.einsum('nd,na,nb->dab', dv, kfu, kfu)
        dKuuinv_v = 2 * np.einsum('dab,dac->bc', KuuinvSmmd, dBhat) \
            - np.sum(dBhat, axis=0)
        dKuuinv = dKuuinv_m + dKuuinv_v + dKuuinv_via_phi
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

    def compute_cavity(self, alpha):
        """compute the leave one out moments and a few other things for training
        
        Args:
            alpha (float, optional): Description
        """
        beta = (self.N - alpha) * 1.0 / self.N
        Dout = self.Dout
        Kuu = self.Kuu
        Kuuinv = self.Kuuinv
        if self.nat_param:
            self.Suhatinv = Kuuinv + beta * self.theta_1
            self.Suhat = np.linalg.inv(self.Suhatinv)
            self.muhat = np.einsum('dab,db->da', self.Suhat, beta * self.theta_2)
        else:
            data_f_1 = self.Suinv - Kuuinv
            data_f_2 = np.einsum('dab,db->da', self.Suinv, self.mu)
            cav_f_1 = Kuuinv + beta * data_f_1
            cav_f_2 = beta * data_f_2
            self.Suhatinv = cav_f_1
            self.Suhat = np.linalg.inv(self.Suhatinv)
            self.muhat = np.einsum('dab,db->da', self.Suhat, cav_f_2)

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

    def compute_cav_grad_u(self, dmu, dSu, alpha):
        # return grads wrt u params and Kuuinv
        triu_ind = np.triu_indices(self.M)
        diag_ind = np.diag_indices(self.M)
        beta = (self.N - alpha) * 1.0 / self.N
        if self.nat_param:
            dSu_via_m = np.einsum('da,db->dab', dmu, beta * self.theta_2)
            dSu += dSu_via_m
            dSuinv = - np.einsum('dab,dbc,dce->dae', self.Suhat, dSu, self.Suhat)
            dKuuinv = np.sum(dSuinv, axis=0)
            dtheta1 = beta * dSuinv
            deta2 = beta * np.einsum('dab,db->da', self.Suhat, dmu)
        else:
            Suhat = self.Suhat
            Suinv = self.Suinv
            mu = self.mu
            data_f_2 = np.einsum('dab,db->da', Suinv, mu)
            dSuhat_via_mhat = np.einsum('da,db->dab', dmu, beta * data_f_2)
            dSuhat = dSu + dSuhat_via_mhat
            dmuhat = dmu
            dSuhatinv = - np.einsum('dab,dbc,dce->dae', Suhat, dSuhat, Suhat)
            dSuinv_1 = beta * dSuhatinv
            Suhatdmu = np.einsum('dab,db->da', Suhat, dmuhat)
            dSuinv = dSuinv_1 + beta * np.einsum('da,db->dab', Suhatdmu, mu)
            dtheta1 = - np.einsum('dab,dbc,dce->dae', Suinv, dSuinv, Suinv)
            deta2 = beta * np.einsum('dab,db->da', Suinv, Suhatdmu)
            dKuuinv = (1 - beta) / beta * np.sum(dSuinv_1, axis=0)

        dtheta1T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum('dab,dbc->dac', self.theta_1_R, dtheta1 + dtheta1T)
        deta1_R = np.zeros([self.Dout, self.M * (self.M + 1) / 2])
        for d in range(self.Dout):
            dtheta1_R_d = dtheta1_R[d, :, :]
            theta1_R_d = self.theta_1_R[d, :, :]
            dtheta1_R_d[diag_ind] = dtheta1_R_d[diag_ind] * theta1_R_d[diag_ind]
            dtheta1_R_d = dtheta1_R_d[triu_ind]
            deta1_R[d, :] = dtheta1_R_d.reshape((dtheta1_R_d.shape[0], ))

        return deta1_R, deta2, dKuuinv


class SGPR(Base_SGPR):
    """Summary
    
    Attributes:
        sgp_layer (TYPE): Description
    """

    def __init__(self, x_train, y_train, no_pseudo, 
        lik='Gaussian', nat_param=True):
        """Summary
        
        Args:
            x_train (TYPE): Description
            y_train (TYPE): Description
            no_pseudo (TYPE): Description
            lik (str, optional): Description
        """
        super(SGPR, self).__init__(x_train, y_train, no_pseudo, lik)
        self.sgp_layer = SGP_Layer(self.N, self.Din, self.Dout, self.M, nat_param)
        self.nat_param = nat_param

    @profile
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
            xb = self.x_train
            yb = self.y_train
        else:
            idxs = np.random.choice(N, mb_size, replace=False)
            xb = self.x_train[idxs, :]
            yb = self.y_train[idxs, :]
        batch_size = yb.shape[0]
        scale_logZ = - N * 1.0 / batch_size / alpha
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


class SGPLVM(Base_SGPLVM):
    """Summary
    
    Attributes:
        sgp_layer (TYPE): Description
    """

    def __init__(self, y_train, hidden_size, no_pseudo,
                 lik='Gaussian', prior_mean=0, prior_var=1, nat_param=True):
        """Summary
        
        Args:
            y_train (TYPE): Description
            hidden_size (TYPE): Description
            no_pseudo (TYPE): Description
            lik (str, optional): Description
            prior_mean (int, optional): Description
            prior_var (int, optional): Description
        """
        super(SGPLVM, self).__init__(
            y_train, hidden_size, no_pseudo, 
            lik, prior_mean, prior_var, nat_param)
        self.sgp_layer = SGP_Layer(self.N, self.Din, self.Dout, self.M, nat_param)
        
    @profile
    def objective_function(self, params, mb_size, alpha=1.0, prop_mode=PROP_MM):
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
        scale_logZ = - N * 1.0 / batch_size / alpha
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1

        # update model with new hypers
        self.update_hypers(params)
        # compute cavity
        self.sgp_layer.compute_cavity(alpha)
        mcav, vcav = self.get_cavity_x(alpha, idxs)
        mpost, vpost = self.get_posterior_x(idxs)
        # for testing only
        # prop_mode = PROP_MC
        if prop_mode == PROP_MM:
            # propagate x cavity forward
            mout, vout, psi1, psi2 = sgp_layer.forward_prop_thru_cav(mcav, vcav)
            # compute logZ and gradients
            logZ, dm, dv = self.lik_layer.compute_log_Z(mout, vout, yb, alpha)
            logZ_scale = scale_logZ * logZ
            dm_scale = scale_logZ * dm
            dv_scale = scale_logZ * dv
            sgp_grad_hyper, sgp_grad_input = sgp_layer.backprop_grads_lvm_mm(
                mout, vout, dm_scale, dv_scale, psi1, psi2, mcav, vcav, alpha)
            lik_grad_hyper = self.lik_layer.backprop_grads(
                mout, vout, dm, dv, alpha, scale_logZ)
        elif prop_mode == PROP_MC:
            # propagate x cavity forward
            res, res_s = sgp_layer.forward_prop_thru_cav(mcav, vcav, PROP_MC)
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
                dx, mcav, vcav, eps)
            lik_grad_hyper = lik_layer.backprop_grads(
                m, v, dm, dv, alpha, scale_logZ)
        else:
            raise NotImplementedError('propagation mode not implemented')

        grad_all = {}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute phi_x and gradients
        phi_prior_x, _, _ = self.compute_phi_x(self.prior_mean, self.prior_var)
        phi_prior_x *= self.N * self.Din
        phi_cav_x, dmcav, dvcav = self.compute_phi_x(mcav, vcav)
        phi_post_x, dmpost, dvpost = self.compute_phi_x(mpost, vpost)
        scale_cav = - self.N * 1.0 / batch_size / alpha
        scale_post = - self.N * 1.0 / batch_size * (1.0 - 1.0 / alpha)
        x_contrib = phi_prior_x + scale_cav * phi_cav_x + scale_post * phi_post_x

        dmcav = scale_cav * dmcav + sgp_grad_input['mx']
        dvcav = scale_cav * dvcav + sgp_grad_input['vx'] 
        dmpost = scale_post * dmpost
        dvpost = scale_post * dvpost
        grads_x_cav = self.compute_cav_grad_x(dmcav, dvcav, mcav, vcav, alpha, idxs)
        grads_x_post = self.compute_posterior_grad_x(dmpost, dvpost, idxs)
        for key in grads_x_cav.keys():
            grad_all[key] = grads_x_cav[key] + grads_x_post[key]

        # compute objective
        sgp_contrib = self.sgp_layer.compute_phi(alpha)
        energy = logZ_scale + x_contrib + sgp_contrib

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def compute_cav_grad_x(self, dm, dv, m, v, alpha, idxs):
        t1 = m / v
        t2 = 1 / v
        dt1 = (1.0 - alpha) * dm / t2
        dt2 = (1.0 - alpha) * (-dm * t1 / t2**2 - dv / t2**2)
        if self.nat_param:
            dt2 *= 2*self.factor_x2[idxs, :]
        else:
            mpost = self.factor_x1[idxs, :]
            vpost = self.factor_x2[idxs, :]
            t1 = mpost / vpost
            t2 = 1 / vpost
            dm = dt1 / vpost
            dv = - dt1 * mpost / vpost**2 - dt2 / vpost ** 2
            dt1 = dm
            dt2 = dv * 2 * self.factor_x2[idxs, :]
            
        d1 = np.zeros_like(self.factor_x1)
        d2 = np.zeros_like(self.factor_x2)
        d1[idxs, :] = dt1
        d2[idxs, :] = dt2
        return {'x1': d1, 'x2': d2}

    def get_cavity_x(self, alpha, idxs=None):
        if idxs is None:
            idxs = np.arange(self.N)
        if self.nat_param:
            t01 = self.prior_x1
            t02 = self.prior_x2
            t11 = self.factor_x1[idxs, :]
            t12 = self.factor_x2[idxs, :]
            cav_t1 = t01 + (1.0 - alpha) * t11
            cav_t2 = t02 + (1.0 - alpha) * t12
            vx = 1.0 / cav_t2
            mx = cav_t1 / cav_t2
        else:
            mpost = self.factor_x1[idxs, :]
            vpost = self.factor_x2[idxs, :]
            t1 = mpost / vpost
            t2 = 1 / vpost
            cav_t1 = self.prior_x1 + (t1 - self.prior_x1) * (1 - alpha)
            cav_t2 = self.prior_x2 + (t2 - self.prior_x2) * (1 - alpha)
            vx = 1.0 / cav_t2
            mx = cav_t1 / cav_t2
        return mx, vx

    def compute_phi_x(self, mx, vx):
        phi = np.sum(0.5 * (mx**2 / vx + np.log(vx)))
        dmx = mx / vx
        dvx = 0.5 * (- mx**2 / vx**2 + 1 / vx)
        return phi, dmx, dvx


class SDGPR(Base_SDGPR):
    """Summary
    
    Attributes:
        sgp_layers (list): Description
    """

    def __init__(self, x_train, y_train, no_pseudos, hidden_sizes, lik='Gaussian'):
        """Summary
        
        Args:
            x_train (TYPE): Description
            y_train (TYPE): Description
            no_pseudos (TYPE): Description
            hidden_sizes (TYPE): Description
            lik (str, optional): Description
        """
        super(SDGPR, self).__init__(x_train, y_train, no_pseudos, hidden_sizes, lik)
        self.sgp_layers = []
        for i in range(self.L):
            Din_i = self.size[i]
            Dout_i = self.size[i + 1]
            M_i = self.Ms[i]
            self.sgp_layers.append(SGP_Layer(self.N, Din_i, Dout_i, M_i))

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
            xb = self.x_train
            yb = self.y_train
        else:
            idxs = np.random.choice(N, mb_size, replace=False)
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

        # propagate x cavity forward
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
                    mout[i - 1], vout[i - 1])
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
        for i in range(self.L - 1, -1, -1):
            layer = self.sgp_layers[i]
            if i == 0:
                grad_hyper = layer.backprop_grads_reg(
                    mout[i], vout[i], dmi, dvi, psi1[i], xb, alpha)
            else:
                grad_hyper, grad_input = layer.backprop_grads_lvm_mm(
                    mout[i], vout[i], dmi, dvi, psi1[i], psi2[i],
                    mout[i - 1], vout[i - 1], alpha)
                dmi, dvi = grad_input['mx'], grad_input['vx']
            grad_list.insert(0, grad_hyper)

        lik_grad_hyper = self.lik_layer.backprop_grads(
            mout[-1], vout[-1], dm, dv, alpha, scale_logZ)

        grad_all = {}
        for i, grad in enumerate(grad_list):
            for key in grad.keys():
                grad_all[key + '_%d' % i] = grad[key]

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


class SGPSSM(Base_SGPSSM):
    """Summary
    
    Attributes:
        dyn_layer (TYPE): Description
        emi_layer (TYPE): Description
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
            x_control (None, optional): Description
            gp_emi (bool, optional): Description
            control_to_emi (bool, optional): Description
        """
        super(SGPSSM, self).__init__(
            y_train, hidden_size, no_pseudo, lik, prior_mean, prior_var,
            x_control, gp_emi, control_to_emi)
        self.dyn_layer = SGP_Layer(
            self.N - 1, self.Din + self.Dcon_dyn, self.Din, self.M)
        if gp_emi:
            self.emi_layer = SGP_Layer(
                self.N, self.Din + self.Dcon_emi, self.Dout, self.M)

    @profile
    def objective_function(self, params, mb_size, alpha=1.0, prop_mode=PROP_MM):
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
        dyn_layer = self.dyn_layer
        emi_layer = self.emi_layer
        if self.gp_emi:
            lik_layer = self.lik_layer
        # TODO: deal with minibatch here
        if mb_size >= N:
            dyn_idxs = np.arange(0, N - 1)
            emi_idxs = np.arange(0, N)
            yb = self.y_train
        else:
            start_idx = np.random.randint(0, N - mb_size)
            # start_idx = 0
            # start_idx = N-mb_size-1
            end_idx = start_idx + mb_size
            emi_idxs = np.arange(start_idx, end_idx)
            dyn_idxs = np.arange(start_idx, end_idx - 1)
            yb = self.y_train[emi_idxs, :]
        batch_size_dyn = dyn_idxs.shape[0]
        scale_logZ_dyn = - (N - 1) * 1.0 / batch_size_dyn / alpha
        batch_size_emi = emi_idxs.shape[0]
        scale_logZ_emi = - N * 1.0 / batch_size_emi / alpha

        # update model with new hypers
        self.update_hypers(params)
        dyn_layer.compute_cavity(alpha)
        if self.gp_emi:
            emi_layer.compute_cavity(alpha)
        cav_m, cav_v, cav_1, cav_2 = self.compute_cavity_x(alpha)
        # compute cavity factors for the latent variables
        idxs_prev = dyn_idxs + 1
        cav_t_m, cav_t_v = cav_m[idxs_prev, :], cav_v[idxs_prev, :]
        idxs_next = dyn_idxs
        cav_tm1_m, cav_tm1_v = cav_m[idxs_next, :], cav_v[idxs_next, :]
        if self.Dcon_dyn > 0:
            cav_tm1_mc = np.hstack((cav_tm1_m, self.x_control[idxs_next, :]))
            cav_tm1_vc = np.hstack(
                (cav_tm1_v, np.zeros((batch_size_dyn, self.Dcon_dyn))))
        else:
            cav_tm1_mc, cav_tm1_vc = cav_tm1_m, cav_tm1_v

        idxs_up = emi_idxs
        cav_up_m, cav_up_v = cav_m[idxs_up, :], cav_v[idxs_up, :]
        if self.Dcon_emi > 0:
            cav_up_mc = np.hstack((cav_up_m, self.x_control[idxs_up, :]))
            cav_up_vc = np.hstack(
                (cav_up_v, np.zeros((batch_size_emi, self.Dcon_emi))))
        else:
            cav_up_mc, cav_up_vc = cav_up_m, cav_up_v

        if prop_mode == PROP_MM:
            # deal with the transition/dynamic factors here
            mprop, vprop, psi1, psi2 = dyn_layer.forward_prop_thru_cav(
                cav_tm1_mc, cav_tm1_vc)
            logZ_dyn, dmprop, dvprop, dmt, dvt, dsn = self.compute_transition_tilted(
                mprop, vprop, cav_t_m, cav_t_v, alpha, scale_logZ_dyn)
            sgp_grad_hyper, sgp_grad_input = dyn_layer.backprop_grads_lvm_mm(
                mprop, vprop, dmprop, dvprop,
                psi1, psi2, cav_tm1_mc, cav_tm1_vc, alpha)

            if self.gp_emi:
                # deal with the emission factors here
                mout, vout, psi1, psi2 = emi_layer.forward_prop_thru_cav(
                    cav_up_mc, cav_up_vc)
                logZ_emi, dm, dv = lik_layer.compute_log_Z(
                    mout, vout, yb, alpha)
                logZ_emi = scale_logZ_emi * logZ_emi
                dm_scale = scale_logZ_emi * dm
                dv_scale = scale_logZ_emi * dv
                emi_grad_hyper, emi_grad_input = emi_layer.backprop_grads_lvm_mm(
                    mout, vout, dm_scale, dv_scale, psi1, psi2,
                    cav_up_mc, cav_up_vc, alpha)
                lik_grad_hyper = lik_layer.backprop_grads(
                    mout, vout, dm, dv, alpha, scale_logZ_emi)
        elif prop_mode == PROP_MC:
            # deal with the transition/dynamic factors here
            res, res_s = dyn_layer.forward_prop_thru_cav(
                cav_tm1_mc, cav_tm1_vc, PROP_MC)
            m, v, kfu, x, eps = res[0], res[1], res[2], res[3], res[4]
            m_s, v_s, kfu_s, x_s, eps_s = (
                res_s[0], res_s[1], res_s[2], res_s[3], res_s[4])
            logZ_dyn, dmprop, dvprop, dmt, dvt, dsn = self.compute_transition_tilted(
                m, v, cav_t_m, cav_t_v, alpha, scale_logZ_dyn)
            sgp_grad_hyper, dx = dyn_layer.backprop_grads_lvm_mc(
                m_s, v_s, dmprop, dvprop, kfu_s, x_s, alpha)
            sgp_grad_input = dyn_layer.backprop_grads_reparam(
                dx, cav_tm1_mc, cav_tm1_vc, eps)
            if self.gp_emi:
                # deal with the emission factors here
                res, res_s = emi_layer.forward_prop_thru_cav(
                    cav_up_mc, cav_up_vc, PROP_MC)
                m, v, kfu, x, eps = res[0], res[1], res[2], res[3], res[4]
                m_s, v_s, kfu_s, x_s, eps_s = (
                    res_s[0], res_s[1], res_s[2], res_s[3], res_s[4])
                # compute logZ and gradients
                logZ_emi, dm, dv = lik_layer.compute_log_Z(
                    m, v, yb, alpha)
                logZ_emi = scale_logZ_emi * logZ_emi
                dm_scale = scale_logZ_emi * dm
                dv_scale = scale_logZ_emi * dv
                emi_grad_hyper, dx = emi_layer.backprop_grads_lvm_mc(
                    m_s, v_s, dm_scale, dv_scale, kfu_s, x_s, alpha)
                emi_grad_input = emi_layer.backprop_grads_reparam(
                    dx, cav_up_mc, cav_up_vc, eps)
                lik_grad_hyper = lik_layer.backprop_grads(
                    m, v, dm, dv, alpha, scale_logZ_emi)
        else:
            raise NotImplementedError('propgation mode not implemented')

        if not self.gp_emi:
            logZ_emi, emi_grad_input, emi_grad_hyper = emi_layer.compute_emission_tilted(
                cav_up_mc, cav_up_vc, alpha, scale_logZ_emi, emi_idxs)

        # collect emission and GP hyperparameters
        grad_all = {'sn': dsn}
        for key in sgp_grad_hyper.keys():
            grad_all[key + '_dynamic'] = sgp_grad_hyper[key]
        for key in emi_grad_hyper.keys():
            grad_all[key + '_emission'] = emi_grad_hyper[key]
        if self.gp_emi:
            for key in lik_grad_hyper.keys():
                grad_all[key + '_emission'] = lik_grad_hyper[key]

        dmcav_up = emi_grad_input['mx'][:, :self.Din]
        dvcav_up = emi_grad_input['vx'][:, :self.Din]
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
            dmcav_prev, dvcav_prev, dmcav_next, dvcav_next, emi_idxs)

        grads_x = {}
        for key in ['x_factor_1', 'x_factor_2']:
            grads_x[key] = grads_x_via_post[key] + \
                grads_x_via_cavity[key] + grads_x_via_logZ[key]

        for key in grads_x.keys():
            grad_all[key] = grads_x[key]

        # compute objective
        dyn_contrib = dyn_layer.compute_phi(alpha)
        if self.gp_emi:
            emi_contrib = emi_layer.compute_phi(alpha)
        else:
            emi_contrib = 0
        phi_prior_x = self.compute_phi_prior_x()
        phi_poste_x = self.compute_phi_posterior_x(alpha)
        phi_cavity_x = self.compute_phi_cavity_x(alpha)
        x_contrib = phi_prior_x + phi_poste_x + phi_cavity_x
        energy = logZ_dyn + logZ_emi + x_contrib + dyn_contrib + emi_contrib
        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def compute_posterior_grad_x(self, alpha):
        """Summary
        
        Args:
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        dpost_1 = post_1 / post_2
        dpost_2 = - 0.5 * post_1**2 / post_2**2 - 0.5 / post_2
        scale_x_post = - (1.0 - 1.0 / alpha) * np.ones((self.N, 1))
        scale_x_post[0:self.N - 1] = scale_x_post[0:self.N - 1] + 1.0 / alpha
        scale_x_post[1:self.N] = scale_x_post[1:self.N] + 1.0 / alpha
        grads_x_1 = scale_x_post * dpost_1
        grads_x_2 = scale_x_post * dpost_2
        grads = {}
        grads['x_factor_1'] = 3.0 * grads_x_1
        grads['x_factor_2'] = 6.0 * grads_x_2 * self.x_factor_2
        grads['x_factor_1'][[0, -1], :] = 2.0 * grads_x_1[[0, -1], :]
        grads['x_factor_2'][[0, -1], :] = (
            4.0 * grads_x_2[[0, -1], :] * self.x_factor_2[[0, -1], :])
        return grads

    def compute_logZ_grad_x(
            self, alpha, cav_1, cav_2, dmcav_up, dvcav_up,
            dmcav_prev, dvcav_prev, dmcav_next, dvcav_next, emi_idxs):
        """Summary
        
        Args:
            alpha (TYPE): Description
            cav_1 (TYPE): Description
            cav_2 (TYPE): Description
            dmcav_up (TYPE): Description
            dvcav_up (TYPE): Description
            dmcav_prev (TYPE): Description
            dvcav_prev (TYPE): Description
            dmcav_next (TYPE): Description
            dvcav_next (TYPE): Description
            emi_idxs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        grads_x_1 = np.zeros_like(cav_1)
        grads_x_2 = np.zeros_like(grads_x_1)

        grads_up_1 = dmcav_up / cav_2[emi_idxs, :]
        grads_up_2 = (- dmcav_up * cav_1[emi_idxs, :] / cav_2[emi_idxs, :]**2
                      - dvcav_up / cav_2[emi_idxs, :]**2)
        grads_x_1[emi_idxs, :] = grads_up_1
        grads_x_2[emi_idxs, :] = grads_up_2

        idxs = np.arange(emi_idxs[0] + 1, emi_idxs[-1] + 1)
        grads_prev_1 = dmcav_prev / cav_2[idxs, :]
        grads_prev_2 = (- dmcav_prev * cav_1[idxs, :] / cav_2[idxs, :]**2
                        - dvcav_prev / cav_2[idxs, :]**2)
        grads_x_1[idxs, :] += grads_prev_1
        grads_x_2[idxs, :] += grads_prev_2

        idxs = np.arange(emi_idxs[0], emi_idxs[-1])
        grads_next_1 = dmcav_next / cav_2[idxs, :]
        grads_next_2 = (- dmcav_next * cav_1[idxs, :] / cav_2[idxs, :]**2
                        - dvcav_next / cav_2[idxs, :]**2)
        grads_x_1[idxs, :] += grads_next_1
        grads_x_2[idxs, :] += grads_next_2

        scale_x_cav = (3.0 - alpha) * np.ones((self.N, 1))
        scale_x_cav[0] = 2.0 - alpha
        scale_x_cav[-1] = 2.0 - alpha
        grad_1 = grads_x_1 * scale_x_cav
        grad_2 = grads_x_2 * scale_x_cav
        grads = {}
        grads['x_factor_1'] = grad_1
        grads['x_factor_2'] = grad_2 * 2 * self.x_factor_2
        return grads

    def compute_cavity_grad_x(self, alpha, cav_1, cav_2):
        """Summary
        
        Args:
            alpha (TYPE): Description
            cav_1 (TYPE): Description
            cav_2 (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        scale = -1.0 / alpha
        dcav_1 = cav_1 / cav_2
        dcav_2 = - 0.5 * cav_1**2 / cav_2**2 - 0.5 / cav_2
        scale_x_cav = scale * np.ones((self.N, 1))
        scale_x_cav[0:self.N - 1] = scale_x_cav[0:self.N - 1] + scale
        scale_x_cav[1:self.N] = scale_x_cav[1:self.N] + scale
        grads_x_1 = scale_x_cav * dcav_1
        grads_x_2 = scale_x_cav * dcav_2

        scale_x_cav = (3.0 - alpha) * np.ones((self.N, 1))
        scale_x_cav[0] = 2.0 - alpha
        scale_x_cav[-1] = 2.0 - alpha
        grad_1 = grads_x_1 * scale_x_cav
        grad_2 = grads_x_2 * scale_x_cav
        grads = {}
        grads['x_factor_1'] = grad_1
        grads['x_factor_2'] = grad_2 * 2 * self.x_factor_2
        return grads

    def compute_transition_tilted(self, m_prop, v_prop, m_t, v_t, alpha, scale):
        """Summary
        
        Args:
            m_prop (TYPE): Description
            v_prop (TYPE): Description
            m_t (TYPE): Description
            v_t (TYPE): Description
            alpha (TYPE): Description
            scale (TYPE): Description
        
        Returns:
            TYPE: Description
        
        Raises:
            RuntimeError: Description
        """
        sn2 = np.exp(2 * self.sn)
        v_sum = v_t + v_prop + sn2 / alpha
        m_diff = m_t - m_prop
        exp_term = -0.5 * m_diff**2 / v_sum
        # const_term = -0.5 * np.log(2 * np.pi * v_sum)
        # alpha_term = 0.5 * (1 - alpha) * np.log(2 *
        #                                         np.pi * sn2) - 0.5 * np.log(alpha)
        alpha_term = - 0.5 * alpha * np.log(2 * np.pi * sn2)
        const_term = - 0.5 * np.log(1 + alpha * (v_t + v_prop) / sn2)
        logZ = exp_term + const_term + alpha_term

        if m_prop.ndim == 2:
            logZ = scale * np.sum(logZ)
            dvt = scale * (-0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2)
            dvprop = dvt
            dmt = scale * (-m_diff / v_sum)
            dmprop = -dmt
            dv_sum = np.sum(dvt)
            dsn = dv_sum * 2 * sn2 / alpha + \
                scale * m_prop.shape[0] * self.Din * (1 - alpha)
        elif m_prop.ndim == 3:
            logZ_max = np.max(logZ, axis=0)
            exp_term = np.exp(logZ - logZ_max)
            sumexp = np.sum(exp_term, axis=0)
            logZ_lse = logZ_max + np.log(sumexp)
            logZ_lse -= np.log(m_prop.shape[0])
            logZ = scale * np.sum(logZ_lse)
            dlogZ = scale * exp_term / sumexp
            dlogZ_dm = dlogZ * m_diff / v_sum
            dlogZ_dv = dlogZ * (-0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2)
            dmt = -np.sum(dlogZ_dm, axis=0)
            dmprop = dlogZ_dm
            dvt = np.sum(dlogZ_dv, axis=0)
            dvprop = dlogZ_dv
            dv_sum = np.sum(dlogZ_dv)
            dsn = dv_sum * 2 * sn2 / alpha + \
                scale * m_prop.shape[1] * self.Din * (1 - alpha)
        else:
            raise RuntimeError('invalid ndim, ndim=%d' % mout.ndim)

        return logZ, dmprop, dvprop, dmt, dvt, dsn

    def compute_cavity_x(self, alpha):
        """Summary
        
        Args:
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        cav_1 = self.x_post_1 - alpha * self.x_factor_1
        cav_2 = self.x_post_2 - alpha * self.x_factor_2
        return cav_1 / (cav_2 + 1e-16), 1.0 / (cav_2 + 1e-16), cav_1, cav_2

    def compute_phi_prior_x(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        t1 = self.x_prior_1
        t2 = self.x_prior_2
        m = t1 / t2
        v = 1.0 / t2
        return 0.5 * self.Din * (m**2 / v + np.log(v))

    def compute_phi_posterior_x(self, alpha):
        """Summary
        
        Args:
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        post_1 = self.x_post_1
        post_2 = self.x_post_2
        phi_post = 0.5 * (post_1**2 / post_2 - np.log(post_2))
        scale_x_post = - (1.0 - 1.0 / alpha) * np.ones((self.N, 1))
        scale_x_post[0:self.N - 1] = scale_x_post[0:self.N - 1] + 1 / alpha
        scale_x_post[1:self.N] = scale_x_post[1:self.N] + 1 / alpha
        return np.sum(scale_x_post * phi_post)

    def compute_phi_cavity_x(self, alpha):
        """Summary
        
        Args:
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        scale = -1.0 / alpha
        cav_1 = self.x_post_1 - alpha * self.x_factor_1
        cav_2 = self.x_post_2 - alpha * self.x_factor_2
        phi_cav = 0.5 * (cav_1**2 / cav_2 - np.log(cav_2))

        scale_x_cav = scale * np.ones((self.N, 1))
        scale_x_cav[0:self.N - 1] = scale_x_cav[0:self.N - 1] + scale
        scale_x_cav[1:self.N] = scale_x_cav[1:self.N] + scale
        phi_cav_sum = np.sum(scale_x_cav * phi_cav)
        return phi_cav_sum


class SDGPR_H(Base_Model):
    """Deep GP regression model with inference for hidden variables
    
    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        h_factor_1 (list): Description
        h_factor_2 (list): Description
        L (TYPE): Description
        lik_layer (list): Description
        Ms (TYPE): Description
        N (TYPE): Description
        sgp_layers (list): Description
        size (TYPE): Description
        sn (TYPE): Description
        updated (bool): Description
        x_train (TYPE): Description
    """

    def __init__(self, x_train, y_train, no_pseudos, hidden_sizes, lik='Gaussian'):
        """Summary
        
        Args:
            x_train (TYPE): Description
            y_train (TYPE): Description
            no_pseudos (TYPE): Description
            hidden_sizes (TYPE): Description
            lik (str, optional): Description
        
        Raises:
            NotImplementedError: Description
        """
        super(SDGPR_H, self).__init__(y_train)
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
        self.lik_layer = []
        for i in range(L):
            Din_i = self.size[i]
            Dout_i = self.size[i + 1]
            M_i = self.Ms[i]
            self.sgp_layers.append(SGP_Layer(N, Din_i, Dout_i, M_i))

        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout_i)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout_i)
        else:
            raise NotImplementedError('likelihood not implemented')
        # natural params for latent variables
        self.h_factor_1 = []
        self.h_factor_2 = []
        for i in range(L - 1):
            Dout_i = self.size[i + 1]
            self.h_factor_1.append(np.zeros((N, Dout_i)))
            self.h_factor_2.append(np.zeros((N, Dout_i)))

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
        cav_h_1, cav_h_2, cav_h_m, cav_h_v = self.compute_cavity_h(idxs, alpha)
        dmcav_h = []
        dvcav_h = []
        dsn = np.zeros_like(self.sn)
        grad_all = {}
        logZ = 0
        for i in range(self.L - 1):
            layer_i = self.sgp_layers[i]
            if i == 0:
                mprop, vprop, kfu = layer_i.forward_prop_thru_cav(xb)
            else:
                mprop, vprop, psi1, psi2 = layer_i.forward_prop_thru_cav(
                    cav_h_m[i - 1], cav_h_v[i - 1])
            m_i, v_i = cav_h_m[i], cav_h_v[i]
            logZ_i, dmprop, dvprop, dmi, dvi, dsni = self.compute_transition_tilted(
                mprop, vprop, m_i, v_i, alpha, i, scale_logZ)

            if i == 0:
                sgp_grad_hyper = layer_i.backprop_grads_reg(
                    mprop, vprop, dmprop, dvprop, kfu, xb, alpha)
            else:
                sgp_grad_hyper, sgp_grad_input = layer_i.backprop_grads_lvm_mm(
                    mprop, vprop, dmprop, dvprop,
                    psi1, psi2, cav_h_m[i - 1], cav_h_v[i - 1], alpha)
                dmcav_h[i - 1] += sgp_grad_input['mx']
                dvcav_h[i - 1] += sgp_grad_input['vx']

            logZ += logZ_i
            for key in sgp_grad_hyper.keys():
                grad_all[key + '_%d' % i] = sgp_grad_hyper[key]
            dmcav_h.append(dmi)
            dvcav_h.append(dvi)
            dsn[i] = dsni

        # final layer
        i = self.L - 1
        layer_i = self.sgp_layers[-1]
        m_im1, v_im1 = cav_h_m[i - 1], cav_h_v[i - 1]
        mprop, vprop, psi1, psi2 = layer_i.forward_prop_thru_cav(m_im1, v_im1)
        # compute logZ and gradients
        logZ_lik, dmprop, dvprop = self.lik_layer.compute_log_Z(
            mprop, vprop, yb, alpha)
        logZ_scale = scale_logZ * logZ_lik
        dm_scale = scale_logZ * dmprop
        dv_scale = scale_logZ * dvprop
        grad_hyper, grad_input = layer_i.backprop_grads_lvm_mm(
            mprop, vprop, dm_scale, dv_scale, psi1, psi2,
            m_im1, v_im1, alpha)
        lik_grad_hyper = self.lik_layer.backprop_grads(
            mprop, vprop, dmprop, dvprop, alpha, scale_logZ)
        logZ += logZ_scale
        dmcav_h[i - 1] += grad_input['mx']
        dvcav_h[i - 1] += grad_input['vx']
        # compute grads via posterior and cavity
        grads_h = self.compute_grads_hidden(dmcav_h, dvcav_h, alpha)

        for key in grad_hyper.keys():
            grad_all[key + '_%d' % i] = grad_hyper[key]
        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]
        for key in grads_h.keys():
            grad_all[key] = grads_h[key]
        grad_all['sn_hidden'] = dsn

        # compute objective
        sgp_contrib = 0
        for layer in self.sgp_layers:
            sgp_contrib += layer.compute_phi(alpha)
        phi_poste_h = self.compute_phi_posterior_h(alpha)
        phi_cavity_h = self.compute_phi_cavity_h(alpha)
        energy = logZ + sgp_contrib + phi_poste_h + phi_cavity_h

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def compute_grads_hidden(self, dmcav_logZ, dvcav_logZ, alpha):
        """Summary
        
        Args:
            dmcav_logZ (TYPE): Description
            dvcav_logZ (TYPE): Description
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        grads = {}
        for i in range(self.L - 1):
            post_1 = self.h_factor_1[i] * 2
            post_2 = self.h_factor_2[i] * 2
            d1_post = 2 * (post_1 / post_2)
            d2_post = - post_1**2 / post_2**2 - 1 / post_2
            scale_post = - (1.0 - 1.0 / alpha)

            cav_1 = self.h_factor_1[i] * (2.0 - alpha)
            cav_2 = self.h_factor_2[i] * (2.0 - alpha)
            d1_cav = (2 - alpha) * (cav_1 / cav_2)
            d2_cav = (2 - alpha) * (- 0.5 * cav_1**2 / cav_2**2 - 0.5 / cav_2)
            scale_cav = -1.0 / alpha

            d1_logZ = (2 - alpha) * (dmcav_logZ[i] / cav_2)
            d2_logZ = (2 - alpha) * (- dmcav_logZ[i] * cav_1 / cav_2**2
                                     - dvcav_logZ[i] / cav_2**2)

            d1 = d1_logZ + scale_cav * d1_cav + scale_post * d1_post
            d2 = d2_logZ + scale_cav * d2_cav + scale_post * d2_post

            grads['h_factor_1_%d' % i] = d1
            grads['h_factor_2_%d' % i] = 2 * d2 * self.h_factor_2[i]
        return grads

    def compute_phi_cavity_h(self, alpha):
        """Summary
        
        Args:
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        scale = - 1.0 / alpha
        phi_cav = 0
        for i in range(self.L - 1):
            cav_1 = self.h_factor_1[i] * (2.0 - alpha)
            cav_2 = self.h_factor_2[i] * (2.0 - alpha)
            phi_cav_i = 0.5 * (cav_1**2 / cav_2 - np.log(cav_2))
            phi_cav += scale * np.sum(phi_cav_i)
        return phi_cav

    def compute_phi_posterior_h(self, alpha):
        """Summary
        
        Args:
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        scale = - (1.0 - 1.0 / alpha)
        phi_post = 0
        for i in range(self.L - 1):
            post_1 = self.h_factor_1[i] * 2.0
            post_2 = self.h_factor_2[i] * 2.0
            phi_post_i = 0.5 * (post_1**2 / post_2 - np.log(post_2))
            phi_post += scale * np.sum(phi_post_i)
        return phi_post

    def compute_transition_tilted(self, m_prop, v_prop, m_t, v_t, alpha,
                                  layer_ind, scale):
        """Summary
        
        Args:
            m_prop (TYPE): Description
            v_prop (TYPE): Description
            m_t (TYPE): Description
            v_t (TYPE): Description
            alpha (TYPE): Description
            layer_ind (TYPE): Description
            scale (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        sn2 = np.exp(2 * self.sn[layer_ind])
        v_sum = v_t + v_prop + sn2 / alpha
        m_diff = m_t - m_prop
        exp_term = -0.5 * m_diff**2 / v_sum
        const_term = -0.5 * np.log(2 * np.pi * v_sum)
        alpha_term = 0.5 * (1 - alpha) * np.log(2 *
                                                np.pi * sn2) - 0.5 * np.log(alpha)
        logZ = exp_term + const_term + alpha_term
        logZ = np.sum(logZ)

        dvt = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dvprop = -0.5 / v_sum + 0.5 * m_diff**2 / v_sum**2
        dmt = -m_diff / v_sum
        dmprop = m_diff / v_sum

        dv_sum = np.sum(dvt)
        dsn = dv_sum * 2 * sn2 / alpha + \
            m_prop.shape[0] * self.size[layer_ind + 1] * (1 - alpha)
        return scale * logZ, scale * dmprop, scale * dvprop, scale * dmt, scale * dvt, scale * dsn

    def compute_cavity_h(self, idxs, alpha):
        """Summary
        
        Args:
            idxs (TYPE): Description
            alpha (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        cav_h_1, cav_h_2, cav_h_m, cav_h_v = [], [], [], []
        for i in range(self.L - 1):
            cav_h_1_i = self.h_factor_1[i][idxs, :] * (2.0 - alpha)
            cav_h_2_i = self.h_factor_2[i][idxs, :] * (2.0 - alpha)
            cav_h_1.append(cav_h_1_i)
            cav_h_2.append(cav_h_2_i)
            cav_h_m.append(cav_h_1_i / cav_h_2_i)
            cav_h_v.append(1.0 / cav_h_2_i)
        return cav_h_1, cav_h_2, cav_h_m, cav_h_v

    def predict_f(self, inputs):
        """Summary
        
        Args:
            inputs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if not self.updated:
            for layer in self.sgp_layers:
                layer.update_posterior()
            self.updated = True
        for i, layer in enumerate(self.sgp_layers):
            if i == 0:
                mf, vf = layer.forward_prop_thru_post(inputs)
            else:
                mf, vf = layer.forward_prop_thru_post(mf, vf)
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
            for layer in self.sgp_layers:
                layer.update_posterior()
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
        """Summary
        
        Args:
            inputs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        mf, vf = self.predict_f(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def init_hypers(self, y_train):
        """Summary
        
        Args:
            y_train (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        init_params = dict()
        for i in range(self.L):
            if i == 0:
                sgp_params = self.sgp_layers[i].init_hypers(
                    self.x_train,
                    key_suffix='_%d' % i)
            else:
                sgp_params = self.sgp_layers[
                    i].init_hypers(key_suffix='_%d' % i)
            init_params.update(sgp_params)
        lik_params = self.lik_layer.init_hypers()
        init_params.update(lik_params)
        init_params['sn_hidden'] = np.log(0.001) * np.ones(self.L - 1)
        for i in range(self.L - 1):
            init_params['h_factor_1_%d' % i] = np.zeros(
                (self.N, self.size[i + 1]))
            init_params['h_factor_2_%d' % i] = np.log(
                0.01) * np.ones((self.N, self.size[i + 1]))
        return init_params

    def get_hypers(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        params = dict()
        for i in range(self.L):
            sgp_params = self.sgp_layers[i].get_hypers(key_suffix='_%d' % i)
            params.update(sgp_params)
        lik_params = self.lik_layer.get_hypers()
        params.update(lik_params)
        params['sn_hidden'] = self.sn
        for i in range(self.L - 1):
            params['h_factor_1_%d' % i] = self.h_factor_1[i]
            params['h_factor_2_%d' % i] = np.log(self.h_factor_2[i]) / 2
        return params

    def update_hypers(self, params):
        """Summary
        
        Args:
            params (TYPE): Description
        """
        for i in range(self.L):
            self.sgp_layers[i].update_hypers(params, key_suffix='_%d' % i)
        self.lik_layer.update_hypers(params)
        self.sn = params['sn_hidden']
        for i in range(self.L - 1):
            self.h_factor_1[i] = params['h_factor_1_%d' % i]
            self.h_factor_2[i] = np.exp(2 * params['h_factor_2_%d' % i])
