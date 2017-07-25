"""Summary

"""
import numpy as np
import pdb
from scipy.cluster.vq import kmeans2

from utils import *
from kernels import *
from config import *
from base_models import Base_Model, Base_SGP_Layer, Base_SGPR, Base_SGPLVM, Base_SGPSSM
from lik_layers import Gauss_Layer, Probit_Layer, Gauss_Emis


class SGPR_collapsed(Base_Model):
    """Summary
    
    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        ls (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        sf (int): Description
        sn (int): Description
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

        energy /= N
        for key in grad_all.keys():
            grad_all[key] /= N

        return energy, grad_all

    def predict_f(self, inputs, alpha=0.01, marginal=True):
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
        # TODO: implement this
        return fs

    def predict_y(self, inputs, alpha=0.01, marginal=True):
        """Summary
        
        Args:
            inputs (TYPE): Description
            alpha (float, optional): Description
            marginal (bool, optional): Description
        
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
        
        Args:
            y_train (TYPE): Description
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
        """
        self.ls = params['ls']
        self.sf = params['sf']
        self.zu = params['zu']
        self.sn = params['sn']


class SGP_Layer(Base_SGP_Layer):
    """Sparse Gaussian process layer
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

    def compute_KL(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        M, Dout = self.M, self.Dout
        # log det prior
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        logdet_prior = Dout * logdet
        # log det posterior
        (sign, logdet) = np.linalg.slogdet(self.Su)
        logdet_post = np.sum(logdet)
        # trace term
        trace_term = np.sum(self.Kuuinv * self.Splusmm)
        KL = 0.5 * (logdet_prior - logdet_post - Dout * M + trace_term)
        return KL

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
        dSu = np.einsum('ab,dbc,ce->dae', Kuuinv, dB, Kuuinv)
        dmu = 2 * np.einsum('dab,db->da', dSu, mu) \
            + np.einsum('ab,db->da', Kuuinv, dA)
        # add in contribution from the KL term
        dmu += np.einsum('ab,db->da', Kuuinv, mu)
        dSu += 0.5 * (Kuuinv - Suinv)
        deta1_R, deta2, dKuuinv_via_u = self.compute_posterior_grad_u(dmu, dSu)

        # grads wrt Kuu
        dKuuinv_KL = - 0.5 * self.Dout * Kuu + 0.5 * np.sum(Spmm, axis=0)
        dKuuinv_A = np.einsum('da,db->ab', dA, mu)
        KuuinvSmmd = np.einsum('ab,dbc->dac', Kuuinv, Spmm)
        dKuuinv_B = 2 * np.einsum('dab,dac->bc', KuuinvSmmd, dB) \
            - np.sum(dB, axis=0)
        dKuuinv = dKuuinv_A + dKuuinv_B + dKuuinv_via_u + dKuuinv_KL
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
    def backprop_grads_lvm_mc(self, m, v, dm, dv, kfu, x):
        """Summary
        
        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm (TYPE): Description
            dv (TYPE): Description
            kfu (TYPE): Description
            x (TYPE): Description
        
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
        Kuuinv = self.Kuuinv
        Kuu = self.Kuu
        kfuKuuinv = np.dot(kfu, Kuuinv)
        dm = dm.reshape(m.shape)
        dv = dv.reshape(v.shape)
        
        # compute grads wrt kfu
        dkfu_m = np.einsum('nd,dm->nm', dm, self.A)
        dkfu_v = 2 * np.einsum('nd,dab,na->nb', dv, self.B_det, kfu)
        dkfu = dkfu_m + dkfu_v
        dsf2, dls, dzu, dx = compute_kfu_derivatives(
            dkfu, kfu, ls, sf2, x, self.zu, grad_x=True)
        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        # compute grads wrt mean and covariance matrix via log lik exp term
        dmu = np.einsum('nd,nm->dm', dm, kfuKuuinv)
        dSu = np.einsum('na,nd,nb->dab', kfuKuuinv, dv, kfuKuuinv)
        # add in contribution from the KL term
        dmu += np.einsum('ab,db->da', Kuuinv, mu)
        dSu += 0.5 * (Kuuinv - Suinv)
        deta1_R, deta2, dKuuinv_via_u = self.compute_posterior_grad_u(dmu, dSu)

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dA = np.einsum('nd,nm->dm', dm, kfu)
        dKuuinv_m = np.einsum('da,db->ab', dA, mu)
        KuuinvS = np.einsum('ab,dbc->dac', Kuuinv, Su)
        dB = np.einsum('nd,na,nb->dab', dv, kfu, kfu)
        dKuuinv_v = 2 * np.einsum('dab,dac->bc', KuuinvS, dB) - np.sum(dB, axis=0)
        dKuuinv_KL = - 0.5 * self.Dout * Kuu + 0.5 * np.sum(Spmm, axis=0)
        dKuuinv = dKuuinv_m + dKuuinv_v + dKuuinv_via_u + dKuuinv_KL
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

        return grad_hyper, dx

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

        # compute grads wrt mean and covariance matrix via log lik exp term
        dmu = np.einsum('nd,nm->dm', dm, kfuKuuinv)
        dSu = np.einsum('na,nd,nb->dab', kfuKuuinv, dv, kfuKuuinv)
        # add in contribution from the KL term
        dmu += np.einsum('ab,db->da', Kuuinv, mu)
        dSu += 0.5 * (Kuuinv - Suinv)
        deta1_R, deta2, dKuuinv_via_u = self.compute_posterior_grad_u(dmu, dSu)

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dA = np.einsum('nd,nm->dm', dm, kfu)
        dKuuinv_m = np.einsum('da,db->ab', dA, mu)
        KuuinvS = np.einsum('ab,dbc->dac', Kuuinv, Su)
        dB = np.einsum('nd,na,nb->dab', dv, kfu, kfu)
        dKuuinv_v = 2 * np.einsum('dab,dac->bc', KuuinvS, dB) - np.sum(dB, axis=0)
        dKuuinv_KL = - 0.5 * self.Dout * Kuu + 0.5 * np.sum(Spmm, axis=0)
        dKuuinv = dKuuinv_m + dKuuinv_v + dKuuinv_via_u + dKuuinv_KL
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


class SGPR(Base_SGPR):
    """Uncollapsed sparse Gaussian process approximations
    
    Attributes:
        sgp_layer (TYPE): Description
        updated (bool): Description
    """

    def __init__(self, x_train, y_train, no_pseudo, lik='Gaussian', nat_param=True):
        """Summary
        
        Args:
            x_train (TYPE): Description
            y_train (TYPE): Description
            no_pseudo (TYPE): Description
            lik (str, optional): Description
        """
        super(SGPR, self).__init__(x_train, y_train, no_pseudo, lik, nat_param)
        self.sgp_layer = SGP_Layer(self.N, self.Din, self.Dout, self.M, nat_param)
        
    @profile
    def objective_function(self, params, mb_size, alpha='not_used', 
        prop_mode='not_used'):
        """Summary
        
        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            alpha (str, optional): Description
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

        # propagate x forward
        mout, vout, kfu = self.sgp_layer.forward_prop_thru_post(xb, return_info=True)
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

        energy /= N
        for key in grad_all.keys():
            grad_all[key] /= N

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
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
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
        mf, vf = self.sgp_layer.forward_prop_thru_post(inputs)
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
        """
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)


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
            y_train, hidden_size, no_pseudo, lik, 
            prior_mean, prior_var, nat_param)
        self.sgp_layer = SGP_Layer(self.N, self.Din, self.Dout, self.M, nat_param)
        

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
        # compute posterior for x
        mx, vx = self.get_posterior_x(idxs)
        if prop_mode == PROP_MM:
            # propagate x forward
            mout, vout, psi1, psi2 = sgp_layer.forward_prop_thru_post(mx, vx, return_info=True)
            # compute logZ and gradients
            logZ, dm, dv = lik_layer.compute_log_lik_exp(mout, vout, yb)    
            logZ_scale = scale_log_lik * logZ
            dm_scale = scale_log_lik * dm
            dv_scale = scale_log_lik * dv
            sgp_grad_hyper, sgp_grad_input = sgp_layer.backprop_grads_lvm_mm(
                mout, vout, dm_scale, dv_scale, psi1, psi2, mx, vx)
            lik_grad_hyper = lik_layer.backprop_grads_log_lik_exp(
                mout, vout, dm, dv, yb, scale_log_lik)
        elif prop_mode == PROP_MC:
            # propagate x forward
            res, res_s = sgp_layer.forward_prop_thru_post(mx, vx, PROP_MC, return_info=True)
            m, v, kfu, x, eps = res[0], res[1], res[2], res[3], res[4]
            m_s, v_s, kfu_s, x_s, eps_s = (
                res_s[0], res_s[1], res_s[2], res_s[3], res_s[4])
            # compute logZ and gradients
            logZ, dm, dv = lik_layer.compute_log_lik_exp(m, v, yb)
            logZ_scale = scale_log_lik * logZ
            dm_scale = scale_log_lik * dm
            dv_scale = scale_log_lik * dv
            sgp_grad_hyper, dx = sgp_layer.backprop_grads_lvm_mc(
                m_s, v_s, dm_scale, dv_scale, kfu_s, x_s)
            sgp_grad_input = sgp_layer.backprop_grads_reparam(
                dx, mx, vx, eps)
            lik_grad_hyper = lik_layer.backprop_grads_log_lik_exp(
                m, v, dm, dv, yb, scale_log_lik)
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
        grads_x = self.compute_posterior_grad_x(dmx, dvx, idxs)
        for key in grads_x.keys():
            grad_all[key] = grads_x[key]

        # compute objective
        sgp_KL_term = self.sgp_layer.compute_KL()
        x_KL_term = scale_x * x_KL_term
        energy = logZ_scale + x_KL_term + sgp_KL_term
        # energy = logZ_scale + x_KL_term
        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        energy /= N
        for key in grad_all.keys():
            grad_all[key] /= N

        return energy, grad_all

    def compute_KL_x(self, mx, vx, m0, v0):
        """Summary
        
        Args:
            mx (TYPE): Description
            vx (TYPE): Description
            m0 (TYPE): Description
            v0 (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        kl = 0.5 * (np.log(v0) - np.log(vx) + (vx + (mx - m0)**2) / v0 - 1)
        kl_sum = np.sum(kl)
        dkl_dmx = (mx - m0) / v0
        dkl_dvx = - 0.5 / vx + 0.5 / v0
        return kl_sum, dkl_dmx, dkl_dvx


class SGPSSM(Base_SGPSSM):
    """Summary
    
    Attributes:
        dyn_layer (TYPE): Description
        emi_layer (TYPE): Description
    """

    def __init__(self, y_train, hidden_size, no_pseudo,
                 lik='Gaussian', prior_mean=0, prior_var=1, x_control=None, 
                 gp_emi=False, control_to_emi=True, nat_param=True):
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
            x_control, gp_emi, control_to_emi, nat_param)
        self.dyn_layer = SGP_Layer(
            self.N - 1, self.Din + self.Dcon_dyn, self.Din, self.M, nat_param)
        if gp_emi:
            self.emi_layer = SGP_Layer(
                self.N, self.Din + self.Dcon_emi, self.Dout, self.M, nat_param)

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
        dyn_layer = self.dyn_layer
        emi_layer = self.emi_layer
        if self.gp_emi:
            lik_layer = self.lik_layer
        if mb_size >= N:
            dyn_idxs = np.arange(0, N - 1)
            emi_idxs = np.arange(0, N)
            yb = self.y_train
        else:
            start_idx = np.random.randint(0, N - mb_size)
            end_idx = start_idx + mb_size
            emi_idxs = np.arange(start_idx, end_idx)
            dyn_idxs = np.arange(start_idx, end_idx - 1)
            yb = self.y_train[emi_idxs, :]
        batch_size_dyn = dyn_idxs.shape[0]
        scale_log_lik_dyn = - (N - 1) * 1.0 / batch_size_dyn
        # scale_log_lik_dyn = 0
        batch_size_emi = emi_idxs.shape[0]
        scale_log_lik_emi = - N * 1.0 / batch_size_emi
        # scale_log_lik_emi = 0

        # update model with new hypers
        self.update_hypers(params)
        # compute posterior
        post_m, post_v = self.get_posterior_x(emi_idxs)
        idxs_prev = dyn_idxs + 1
        post_t_m, post_t_v = post_m[1:, :], post_v[1:, :]
        idxs_next = dyn_idxs
        post_tm1_m, post_tm1_v = post_m[0:-1, :], post_v[0:-1, :]
        if self.Dcon_dyn > 0:
            post_tm1_mc = np.hstack((post_tm1_m, self.x_control[idxs_next, :]))
            post_tm1_vc = np.hstack(
                (post_tm1_v, np.zeros((batch_size_dyn, self.Dcon_dyn))))
        else:
            post_tm1_mc, post_tm1_vc = post_tm1_m, post_tm1_v

        post_up_m, post_up_v = post_m, post_v
        if self.Dcon_emi > 0:
            post_up_mc = np.hstack((post_up_m, self.x_control[emi_idxs, :]))
            post_up_vc = np.hstack(
                (post_up_v, np.zeros((batch_size_emi, self.Dcon_emi))))
        else:
            post_up_mc, post_up_vc = post_up_m, post_up_v

        if prop_mode == PROP_MM:
            # deal with the transition/dynamic factors here
            mprop, vprop, psi1, psi2 = dyn_layer.forward_prop_thru_post(
                post_tm1_mc, post_tm1_vc, return_info=True)
            logZ_dyn, dmprop, dvprop, dmt, dvt, dsn = self.compute_transition_log_lik_exp(
                mprop, vprop, post_t_m, post_t_v, scale_log_lik_dyn)
            sgp_grad_hyper, sgp_grad_input = dyn_layer.backprop_grads_lvm_mm(
                mprop, vprop, dmprop, dvprop,
                psi1, psi2, post_tm1_mc, post_tm1_vc)
            if self.gp_emi:
                # deal with the emission factors here
                mout, vout, psi1, psi2 = emi_layer.forward_prop_thru_post(
                    post_up_mc, post_up_vc, return_info=True)
                logZ_emi, dm, dv = lik_layer.compute_log_lik_exp(
                    mout, vout, yb)
                logZ_emi = scale_log_lik_emi * logZ_emi
                dm_scale = scale_log_lik_emi * dm
                dv_scale = scale_log_lik_emi * dv
                emi_grad_hyper, emi_grad_input = emi_layer.backprop_grads_lvm_mm(
                    mout, vout, dm_scale, dv_scale, 
                    psi1, psi2, post_up_mc, post_up_vc)
                lik_grad_hyper = lik_layer.backprop_grads_log_lik_exp(
                    mout, vout, dm, dv, yb, scale_log_lik_emi)
        elif prop_mode == PROP_MC:
            # TODO
            # deal with the transition/dynamic factors here
            res, res_s = dyn_layer.forward_prop_thru_post(
                post_tm1_mc, post_tm1_vc, PROP_MC, return_info=True)
            m, v, kfu, x, eps = res[0], res[1], res[2], res[3], res[4]
            m_s, v_s, kfu_s, x_s, eps_s = (
                res_s[0], res_s[1], res_s[2], res_s[3], res_s[4])
            logZ_dyn, dmprop, dvprop, dmt, dvt, dsn = self.compute_transition_log_lik_exp(
                m, v, post_t_m, post_t_v, scale_log_lik_dyn)
            sgp_grad_hyper, dx = dyn_layer.backprop_grads_lvm_mc(
                m_s, v_s, dmprop, dvprop, kfu_s, x_s)
            sgp_grad_input = dyn_layer.backprop_grads_reparam(
                dx, post_tm1_mc, post_tm1_vc, eps)
            if self.gp_emi:
                # deal with the emission factors here
                res, res_s = emi_layer.forward_prop_thru_post(
                    post_up_mc, post_up_vc, PROP_MC, return_info=True)
                m, v, kfu, x, eps = res[0], res[1], res[2], res[3], res[4]
                m_s, v_s, kfu_s, x_s, eps_s = (
                    res_s[0], res_s[1], res_s[2], res_s[3], res_s[4])
                # compute logZ and gradients
                logZ_emi, dm, dv = lik_layer.compute_log_lik_exp(m, v, yb)
                logZ_emi = scale_log_lik_emi * logZ_emi
                dm_scale = scale_log_lik_emi * dm
                dv_scale = scale_log_lik_emi * dv
                emi_grad_hyper, dx = emi_layer.backprop_grads_lvm_mc(
                    m_s, v_s, dm_scale, dv_scale, kfu_s, x_s)
                emi_grad_input = emi_layer.backprop_grads_reparam(
                    dx, post_up_mc, post_up_vc, eps)
                lik_grad_hyper = lik_layer.backprop_grads_log_lik_exp(
                    m, v, dm, dv, yb, scale_log_lik_emi)
        else:
            raise NotImplementedError('propgation mode not implemented')

        if not self.gp_emi:
            logZ_emi, emi_grad_input, emi_grad_hyper = emi_layer.compute_emission_log_lik_exp(
                post_up_mc, post_up_vc, scale_log_lik_emi, emi_idxs)

        # collect emission and GP hyperparameters
        grad_all = {'sn': dsn}
        for key in sgp_grad_hyper.keys():
            grad_all[key + '_dynamic'] = sgp_grad_hyper[key]
        for key in emi_grad_hyper.keys():
            grad_all[key + '_emission'] = emi_grad_hyper[key]
        if self.gp_emi:
            for key in lik_grad_hyper.keys():
                grad_all[key + '_emission'] = lik_grad_hyper[key]


        dm_up = emi_grad_input['mx'][:, :self.Din]
        dv_up = emi_grad_input['vx'][:, :self.Din]
        dm_next, dv_next = dmt, dvt
        dm_prev = sgp_grad_input['mx'][:, :self.Din]
        dv_prev = sgp_grad_input['vx'][:, :self.Din]

        # compute entropy
        scale_entropy = - N * 1.0 / batch_size_emi
        # scale_entropy = 0
        x_entrop = batch_size_emi * self.Din * (0.5 + 0.5 * np.log(2*np.pi))
        x_entrop += np.sum(0.5 * np.log(post_v))
        x_entrop *= scale_entropy
        dv_entrop = scale_entropy * 0.5 / post_v

        # TODO ignore x prior term for now
        prior_contrib = 0
        dm0 = 0
        dv0 = 0

        # aggregate gradients for x param
        dm = dm_up
        dm[1:, :] += dm_next
        dm[0:-1, :] += dm_prev
        dv = dv_up + dv_entrop
        dv[1:, :] += dv_next
        dv[0:-1, :] += dv_prev
        grads_x = self.compute_posterior_grad_x(dm, dv, emi_idxs)
        for key in grads_x.keys():
            grad_all[key] = grads_x[key]

        # compute objective
        dyn_KL = dyn_layer.compute_KL()
        if self.gp_emi:
            emi_KL = emi_layer.compute_KL()
        else:
            emi_KL = 0
        energy = logZ_dyn + logZ_emi + x_entrop + dyn_KL + emi_KL
        # print logZ_dyn, logZ_emi, x_entrop, dyn_KL, emi_KL
        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        energy /= N
        for key in grad_all.keys():
            grad_all[key] /= N

        return energy, grad_all

    def compute_transition_log_lik_exp(self, m_prop, v_prop, m_t, v_t, scale):
        """Summary
        
        Args:
            m_prop (TYPE): Description
            v_prop (TYPE): Description
            m_t (TYPE): Description
            v_t (TYPE): Description
            scale (TYPE): Description
        
        Returns:
            TYPE: Description
        
        Raises:
            RuntimeError: Description
        """
        sn2 = np.exp(2 * self.sn)
        if m_prop.ndim == 2:
            term_1 = - 0.5 * np.log(2 * np.pi * sn2)
            term_2 = - 0.5 / sn2 * (m_t**2 + v_t - 2 * m_t * m_prop + m_prop**2 + v_prop)
            logZ = scale * np.sum(term_1 + term_2)
            dmt = - scale / sn2 * (m_t - m_prop)
            dmprop = - dmt
            dvt = - scale * 0.5 / sn2 * np.ones_like(v_t)
            dvprop = dvt
            dsn = scale * np.sum(-1 - 2*term_2)
        elif m_prop.ndim == 3:
            K = m_prop.shape[0]
            term_1 = - 0.5 * np.log(2 * np.pi * sn2)
            term_2 = - 0.5 / sn2 * (m_t**2 + v_t - 2 * m_t * m_prop + m_prop**2 + v_prop)
            logZ = scale * np.sum(term_1 + term_2) / K
            dmprop = scale / sn2 * (m_t - m_prop) / K
            dmt = - np.sum(dmprop, axis=0)
            dvprop = - scale * 0.5 / sn2 * np.ones_like(v_prop) / K
            dvt = np.sum(dvprop, axis=0)
            dsn = scale * np.sum(-1 - 2*term_2) / K
        else:
            raise RuntimeError('invalid ndim, ndim=%d' % mout.ndim)

        return logZ, dmprop, dvprop, dmt, dvt, dsn

