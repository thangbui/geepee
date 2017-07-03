import numpy as np
from scipy import special
from scipy.stats import norm
import pdb

from config import *


class Lik_Layer(object):
    """Summary

    Attributes:
        D (TYPE): Description
        N (TYPE): Description
    """

    def __init__(self, N, D):
        """Summary

        Args:
            N (TYPE): Description
            D (TYPE): Description
        """
        self.N = N
        self.D = D

    def compute_log_Z(self, mout, vout, y, alpha=1.0):
        """Summary

        Args:
            mout (TYPE): Description
            vout (TYPE): Description
            y (TYPE): Description
            alpha (float, optional): Description

        """
        pass

    def backprop_grads(self, mout, vout, dmout, dvout, alpha=1.0, scale=1.0):
        """Summary

        Args:
            mout (TYPE): Description
            vout (TYPE): Description
            dmout (TYPE): Description
            dvout (TYPE): Description
            alpha (float, optional): Description
            scale (float, optional): Description

        Returns:
            TYPE: Description
        """
        return {}

    def compute_log_lik_exp(self, m, v, y):
        pass

    def backprop_grads_log_lik_exp(self, m, v, dm, dv, y, scale=1.0):
        return {}

    def init_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return {}

    def get_hypers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return {}

    def update_hypers(self, params):
        """Summary

        Args:
            params (TYPE): Description

        """
        pass


class Gauss_Layer(Lik_Layer):
    """Summary

    Attributes:
        sn (int): Description
    """

    def __init__(self, N, D):
        """Summary

        Args:
            N (TYPE): Description
            D (TYPE): Description
        """
        super(Gauss_Layer, self).__init__(N, D)
        self.sn = 0

    def compute_log_Z(self, mout, vout, y, alpha=1.0, compute_dm2=False):
        """Summary

        Args:
            mout (TYPE): Description
            vout (TYPE): Description
            y (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description

        Raises:
            RuntimeError: Description
        """
        # real valued data, gaussian lik
        if mout.ndim == 2:
            sn2 = np.exp(2.0 * self.sn)
            vout += sn2 / alpha
            logZ = np.sum(-0.5 * (np.log(2 * np.pi * vout) +
                                  (y - mout)**2 / vout))
            logZ += y.shape[0] * self.D * (0.5 * np.log(2 * np.pi * sn2 / alpha)
                                           - 0.5 * alpha * np.log(2 * np.pi * sn2))
            dlogZ_dm = (y - mout) / vout
            dlogZ_dv = -0.5 / vout + 0.5 * (y - mout)**2 / vout**2
            if compute_dm2:
                dlogZ_dm2 = - 1 / vout
                return logZ, dlogZ_dm, dlogZ_dv, dlogZ_dm2
            else:
                return logZ, dlogZ_dm, dlogZ_dv
        elif mout.ndim == 3:
            sn2 = np.exp(2.0 * self.sn)
            vout += sn2 / alpha
            logZ = -0.5 * (np.log(2 * np.pi * vout) + (y - mout)**2 / vout)
            logZ += (0.5 * np.log(2 * np.pi * sn2 / alpha) -
                     0.5 * alpha * np.log(2 * np.pi * sn2))

            logZ_max = np.max(logZ, axis=0)
            exp_term = np.exp(logZ - logZ_max)
            sumexp = np.sum(exp_term, axis=0)
            logZ_lse = logZ_max + np.log(sumexp)
            logZ_lse -= np.log(mout.shape[0])
            logZ = np.sum(logZ_lse)
            dlogZ = exp_term / sumexp
            dlogZ_dm = dlogZ * (y - mout) / vout
            dlogZ_dv = dlogZ * (-0.5 / vout + 0.5 * (y - mout)**2 / vout**2)
            return logZ, dlogZ_dm, dlogZ_dv
        else:
            raise RuntimeError('invalid ndim, ndim=%d' % mout.ndim)

    def backprop_grads(self, mout, vout, dmout, dvout, alpha=1.0, scale=1.0):
        """Summary

        Args:
            mout (TYPE): Description
            vout (TYPE): Description
            dmout (TYPE): Description
            dvout (TYPE): Description
            alpha (float, optional): Description
            scale (float, optional): Description

        Returns:
            TYPE: Description

        Raises:
            RuntimeError: Description
        """
        sn2 = np.exp(2.0 * self.sn)
        dv_sum = np.sum(dvout)
        if mout.ndim == 2:
            dim_prod = mout.shape[0] * self.D
        elif mout.ndim == 3:
            dim_prod = mout.shape[1] * self.D
        else:
            raise RuntimeError('invalid ndim, ndim=%d' % mout.ndim)
        dsn = dv_sum * 2 * sn2 / alpha + dim_prod * (1 - alpha)
        dsn *= scale
        return {'sn': dsn}

    def compute_log_lik_exp(self, mout, vout, y):
        """Summary

        Args:
            mout (TYPE): Description
            vout (TYPE): Description
            y (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description

        Raises:
            RuntimeError: Description
        """
        sn2 = np.exp(2.0 * self.sn)
        # real valued data, gaussian lik
        if mout.ndim == 2:
            term1 = -0.5 * np.log(2 * np.pi * sn2)
            term2 = -0.5 / sn2 * (y**2 - 2 * y * mout + mout**2 + vout)
            de_dm = 1.0 / sn2 * (y - mout)
            de_dv = -0.5 / sn2 * np.ones_like(vout)
            exptn = term1 + term2
            exptn_sum = np.sum(exptn)

            return exptn_sum, de_dm, de_dv
        elif mout.ndim == 3:
            term1 = - 0.5 * np.log(2*np.pi*sn2)
            term2 = - 0.5 * ((y - mout) ** 2 + vout) / sn2
            sumterm = term1 + term2
            logZ = np.sum(np.mean(sumterm, axis=0))
            dlogZ_dm = (y - mout) / sn2 / mout.shape[0]
            dlogZ_dv = - 0.5 / sn2 * np.ones_like(vout) / mout.shape[0]
            return logZ, dlogZ_dm, dlogZ_dv
        else:
            raise RuntimeError('invalid ndim, ndim=%d' % mout.ndim)

    def backprop_grads_log_lik_exp(self, m, v, dm, dv, y, scale=1.0):
        # real valued data, gaussian lik
        sn2 = np.exp(2.0 * self.sn)
        if m.ndim == 2:
            term1 = -1
            term2 = 1 / sn2 * (y**2 - 2 * y * m + m**2 + v)
            dsn = term1 + term2
            dsn = scale * np.sum(dsn)
            return {'sn': dsn}
        elif m.ndim == 3:
            term1 = - 1
            term2 =  ((y - m) ** 2 + v) / sn2
            dsn = term1 + term2
            dsn = scale * np.sum(dsn) / m.shape[0]
            return {'sn': dsn}
        else:
            raise RuntimeError('invalid ndim, ndim=%d' % mout.ndim)

    def output_probabilistic(self, mf, vf, alpha=1.0):
        """Summary

        Args:
            mf (TYPE): Description
            vf (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description
        """
        return mf, vf + np.exp(2.0 * self.sn) / alpha

    def init_hypers(self, key_suffix=''):
        """Summary

        Args:
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
        self.sn = np.log(0.01)
        return {'sn' + key_suffix: self.sn}

    def get_hypers(self, key_suffix=''):
        """Summary

        Args:
            key_suffix (str, optional): Description

        Returns:
            TYPE: Description
        """
        return {'sn' + key_suffix: self.sn}

    def update_hypers(self, params, key_suffix=''):
        """Summary

        Args:
            params (TYPE): Description
            key_suffix (str, optional): Description

        """
        self.sn = params['sn' + key_suffix]


class Probit_Layer(Lik_Layer):
    """Summary
    """
    __gh_points = None

    def _gh_points(self, T=20):
        """Summary

        Args:
            T (int, optional): Description

        Returns:
            TYPE: Description
        """
        if self.__gh_points is None:
            self.__gh_points = np.polynomial.hermite.hermgauss(T)
        return self.__gh_points

    def compute_log_Z(self, mout, vout, y, alpha=1.0, compute_dm2=False):
        """Summary

        Args:
            mout (TYPE): Description
            vout (TYPE): Description
            y (TYPE): Description
            alpha (float, optional): Description

        Returns:
            TYPE: Description

        Raises:
            RuntimeError: Description
        """
        # binary data probit likelihood
        if mout.ndim == 2:
            if alpha == 1.0:
                t = y * mout / np.sqrt(1 + vout)
                Z = 0.5 * (1 + special.erf(t / np.sqrt(2)))
                eps = 1e-16
                logZ = np.sum(np.log(Z + eps))

                dlogZ_dt = 1 / (Z + eps) 
                dlogZ_dt = dlogZ_dt / np.sqrt(2 * np.pi) * np.exp(-t**2.0 / 2)
                dt_dm = y / np.sqrt(1 + vout)
                dt_dv = -0.5 * y * mout / (1 + vout)**1.5
                dlogZ_dm = dlogZ_dt * dt_dm
                dlogZ_dv = dlogZ_dt * dt_dv

                if compute_dm2:
                    beta = dlogZ_dm / y
                    dlogZ_dm2 = - (beta**2 + mout * y * beta / (1 + vout))
            else:
                gh_x, gh_w = self._gh_points(GH_DEGREE)
                gh_x = gh_x[:, np.newaxis, np.newaxis]
                gh_w = gh_w[:, np.newaxis, np.newaxis]

                ts = gh_x * \
                    np.sqrt(2 * vout[np.newaxis, :, :]) + \
                    mout[np.newaxis, :, :]
                eps = 1e-8
                pdfs = 0.5 * (1 + special.erf(y * ts / np.sqrt(2))) + eps
                Ztilted = np.sum(pdfs**alpha * gh_w, axis=0) / np.sqrt(np.pi)
                logZ = np.sum(np.log(Ztilted))

                a = pdfs**(alpha - 1.0) * np.exp(-ts**2 / 2)
                dZdm = np.sum(gh_w * a, axis=0) * y * \
                    alpha / np.pi / np.sqrt(2)
                dlogZ_dm = dZdm / Ztilted + eps

                dZdv = np.sum(gh_w * (a * gh_x), axis=0) * y * \
                    alpha / np.pi / np.sqrt(2) / np.sqrt(2 * vout)
                dlogZ_dv = dZdv / Ztilted + eps

                if compute_dm2:
                    b = (alpha-1)*pdfs**(alpha-2)*np.exp(-ts**2)/np.sqrt(2*np.pi) \
                        - pdfs**(alpha-1) * y * ts * np.exp(-ts**2/2)
                    dZdm2 = np.sum(gh_w * b, axis=0) * alpha / np.pi / np.sqrt(2)
                    dlogZ_dm2 = -dZdm**2 / Ztilted**2 + dZdm2 / Ztilted + eps

        elif mout.ndim == 3:
            if alpha == 1.0:
                t = y * mout / np.sqrt(1 + vout)
                Z = 0.5 * (1 + special.erf(t / np.sqrt(2)))
                eps = 1e-16
                logZ_term = np.log(Z + eps)
                logZ_max = np.max(logZ_term, axis=0)
                exp_term = np.exp(logZ_term - logZ_max)
                sumexp = np.sum(exp_term, axis=0)
                logZ_lse = logZ_max + np.log(sumexp)
                logZ_lse -= np.log(mout.shape[0])
                logZ = np.sum(logZ_lse)
                dlogZ = exp_term / sumexp
                dlogZ_dt = 1 / (Z + eps) 
                dlogZ_dt = dlogZ_dt / np.sqrt(2 * np.pi) * np.exp(-t**2.0 / 2)
                dt_dm = y / np.sqrt(1 + vout)
                dt_dv = -0.5 * y * mout / (1 + vout)**1.5
                dlogZ_dm = dlogZ * dlogZ_dt * dt_dm
                dlogZ_dv = dlogZ * dlogZ_dt * dt_dv
            else:
                gh_x, gh_w = self._gh_points(GH_DEGREE)
                gh_x = gh_x[:, np.newaxis, np.newaxis, np.newaxis]
                gh_w = gh_w[:, np.newaxis, np.newaxis, np.newaxis]

                ts = gh_x * np.sqrt(2 * vout) + mout
                eps = 1e-16
                pdfs = 0.5 * (1 + special.erf(y * ts / np.sqrt(2))) + eps
                Ztilted = np.sum(pdfs**alpha * gh_w, axis=0) / np.sqrt(np.pi)
                # logZ = np.sum(np.log(Ztilted))
                logZ_term = np.log(Ztilted)
                logZ_max = np.max(logZ_term, axis=0)
                exp_term = np.exp(logZ_term - logZ_max)
                sumexp = np.sum(exp_term, axis=0)
                logZ_lse = logZ_max + np.log(sumexp)
                logZ_lse -= np.log(mout.shape[0])
                logZ = np.sum(logZ_lse)
                dlogZ = exp_term / sumexp

                a = pdfs**(alpha - 1.0) * np.exp(-ts**2 / 2)
                dZdm = np.sum(gh_w * a, axis=0) * y * \
                    alpha / np.pi / np.sqrt(2)
                dlogZ_dm = dlogZ * dZdm / Ztilted + eps

                dZdv = np.sum(gh_w * (a * gh_x), axis=0) * y * \
                    alpha / np.pi / np.sqrt(2) / np.sqrt(2 * vout)
                dlogZ_dv = dlogZ * dZdv / Ztilted + eps
        else:
            raise RuntimeError('invalid ndim, ndim=%d' % mout.ndim)

        if compute_dm2:
            return logZ, dlogZ_dm, dlogZ_dv, dlogZ_dm2
        else:
            return logZ, dlogZ_dm, dlogZ_dv

    def compute_log_lik_exp(self, m, v, y):
        if m.ndim == 2:
            gh_x, gh_w = self._gh_points(GH_DEGREE)
            gh_x = gh_x[:, np.newaxis, np.newaxis]
            gh_w = gh_w[:, np.newaxis, np.newaxis] / np.sqrt(np.pi)
            v_expand = v[np.newaxis, :, :]
            m_expand = m[np.newaxis, :, :]
            ts = gh_x * np.sqrt(2 * v_expand) + m_expand
            logcdfs = norm.logcdf(ts * y)
            prods = gh_w * logcdfs
            loglik = np.sum(prods)

            pdfs = norm.pdf(ts * y)
            cdfs = norm.cdf(ts * y)
            grad_cdfs = y * gh_w * pdfs / cdfs
            dts_dm = 1
            dts_dv = 0.5 * gh_x * np.sqrt(2 / v_expand)
            dm = np.sum(grad_cdfs * dts_dm, axis=0)
            dv = np.sum(grad_cdfs * dts_dv, axis=0)
        else:
            gh_x, gh_w = self._gh_points(GH_DEGREE)
            gh_x = gh_x[:, np.newaxis, np.newaxis, np.newaxis]
            gh_w = gh_w[:, np.newaxis, np.newaxis, np.newaxis] / np.sqrt(np.pi)
            v_expand = v[np.newaxis, :, :, :]
            m_expand = m[np.newaxis, :, :, :]
            ts = gh_x * np.sqrt(2 * v_expand) + m_expand
            logcdfs = norm.logcdf(ts * y)
            prods = gh_w * logcdfs
            prods_mean = np.mean(prods, axis=1)
            loglik = np.sum(prods_mean)

            pdfs = norm.pdf(ts * y)
            cdfs = norm.cdf(ts * y)
            grad_cdfs = y * gh_w * pdfs / cdfs
            dts_dm = 1
            dts_dv = 0.5 * gh_x * np.sqrt(2 / v_expand)
            dm = np.sum(grad_cdfs * dts_dm, axis=0) / m.shape[0]
            dv = np.sum(grad_cdfs * dts_dv, axis=0) / m.shape[0]

        return loglik, dm, dv

    def output_probabilistic(self, mf, vf, alpha=1.0):
        """Summary

        Args:
            mf (TYPE): Description
            vf (TYPE): Description
            alpha (float, optional): Description

        Raises:
            NotImplementedError: Description

        """
        raise NotImplementedError('TODO: return probablity of y=1')


class Gauss_Emis():
    """Summary

    Attributes:
        C (TYPE): Description
        Din (TYPE): Description
        Dout (TYPE): Description
        N (TYPE): Description
        R (TYPE): Description
        y (TYPE): Description
    """

    def __init__(self, y, Dout, Din):
        """Summary

        Args:
            y (TYPE): Description
            Dout (TYPE): Description
            Din (TYPE): Description
        """
        self.y = y
        self.N = y.shape[0]
        self.Dout = Dout
        self.Din = Din
        self.C = np.zeros((Dout, Din))
        self.R = np.zeros(Dout)

    def update_hypers(self, params, key_suffix=''):
        """Summary

        Args:
            params (TYPE): Description
            key_suffix (str, optional): Description
        """
        self.C = params['C' + key_suffix]
        self.R = np.exp(2 * params['R' + key_suffix])

    def init_hypers(self, key_suffix=''):
        """Summary

        Returns:
            TYPE: Description

        Args:
            key_suffix (str, optional): Description
        """
        params = {}
        params['C' + key_suffix] = np.ones((self.Dout, self.Din)) / (self.Dout * self.Din)
        params['R' + key_suffix] = np.log(0.01) * np.ones(self.Dout)
        return params

    def get_hypers(self, key_suffix=''):
        """Summary

        Returns:
            TYPE: Description

        Args:
            key_suffix (str, optional): Description
        """
        params = {
            'C' + key_suffix: self.C,
            'R' + key_suffix: 0.5 * np.log(self.R)}
        return params

    def compute_factor(self, x_cav_m, x_cav_v, alpha):
        """Summary

        Args:
            x_cav_m (TYPE): Description
            x_cav_v (TYPE): Description
            alpha (TYPE): Description

        Returns:
            TYPE: Description
        """
        # this update does not depend on the cavity information and alpha
        CTRinv = self.C.T / self.R
        t2 = np.sum(CTRinv * self.C.T, axis=1)
        t1 = np.dot(CTRinv, self.y.T)
        t2rep = np.tile(t2[np.newaxis, :], [self.N, 1])
        t1 = t1.T
        return t1, t2rep

    def output_probabilistic(self, mf, vf):
        """Summary

        Args:
            mf (TYPE): Description
            vf (TYPE): Description

        Returns:
            TYPE: Description
        """
        my = np.einsum('ab,nb->na', self.C, mf)
        vy_noiseless = np.einsum('ab,nb,bc->nac', self.C, vf, self.C.T)
        vy = vy_noiseless + np.diag(self.R)
        return my, vy_noiseless, vy

    def compute_emission_tilted(self, mx, vx, alpha, scale, idxs=None):
        """Summary

        Args:
            mx (TYPE): Description
            vx (TYPE): Description
            alpha (TYPE): Description
            scale (TYPE): Description
            idxs (None, optional): Description

        Returns:
            TYPE: Description
        """
        if idxs is None:
            idxs = np.arange(self.N)
            assert mx.shape[0] == self.N
            Nb = self.N
        else:
            Nb = mx.shape[0]
            assert mx.shape[0] == idxs.shape[0]
        C = self.C
        R = self.R
        Dout = self.Dout
        Vy = np.diag(R / alpha) + np.einsum('da,na,ab->ndb', C, vx, C.T)
        Ydiff = self.y[idxs, :] - np.einsum('da,na->nd', C, mx)
        VinvY = np.linalg.solve(Vy, Ydiff)
        quad_term = -0.5 * np.sum(Ydiff * VinvY)
        CVC = np.einsum('da,na,ab->ndb', C, vx, C.T)
        CVCR = CVC / R
        I = np.tile(np.eye(Dout)[np.newaxis, :, :], [Nb, 1, 1])
        ICVCR = I + alpha * CVCR
        Vlogdet_term = -0.5 * np.sum(np.linalg.slogdet(ICVCR)[1])
        const_term = - Nb * Dout * 0.5 * alpha * np.log(2 * np.pi)
        Rlogdet_term = - 0.5 * Nb * alpha * np.sum(np.log(R))
        logZ = const_term + Rlogdet_term + Vlogdet_term + quad_term
        # print const_term / alpha, Rlogdet_term / alpha, Vlogdet_term / alpha, quad_term / alpha

        Vyinv = np.linalg.inv(Vy)
        dR = (-0.5 * np.sum(np.diagonal(Vyinv, axis1=1, axis2=2), axis=0)
              + 0.5 * np.sum(VinvY**2, axis=0)) / alpha
        dR += 0.5 * Nb * (1 - alpha) / R
        dR *= 2 * R

        dSigma = -0.5 * Vyinv + 0.5 * np.einsum('na,nb->nab', VinvY, VinvY)
        dmu = VinvY
        dC1 = np.einsum('na,nb->ab', dmu, mx)
        dC2 = 2 * np.einsum('nc,bc,nab->ac', vx, C, dSigma)
        dC = dC1 + dC2

        dmx = np.einsum('na,ab->nb', dmu, C)
        dvx = np.einsum('nab,da,db->nd', dSigma, C.T, C.T)

        emi_grads = {'C': dC * scale, 'R': dR * scale}
        input_grads = {'mx': dmx * scale, 'vx': dvx * scale}
        return logZ * scale, input_grads, emi_grads

    def compute_emission_log_lik_exp(self, mx, vx, scale, idxs=None):
        """Summary

        Args:
            mx (TYPE): Description
            vx (TYPE): Description
            scale (TYPE): Description
            idxs (None, optional): Description

        Returns:
            TYPE: Description
        """
        if idxs is None:
            idxs = np.arange(self.N)
            assert mx.shape[0] == self.N
            Nb = self.N
        else:
            Nb = mx.shape[0]
            assert mx.shape[0] == idxs.shape[0]
        yb = self.y[idxs, :]
        C = self.C
        R = self.R
        Dout = self.Dout
        term1 = - 0.5 * Nb * Dout * np.log(2 * np.pi)
        term2 = - 0.5 * Nb * np.sum(np.log(R))
        Cmx = np.einsum('ab,nb->na', C, mx)
        term3 = - 0.5 * np.sum(np.sum((yb - Cmx)**2, axis=0) / R)
        CRC = np.dot(C.T, np.dot(np.diag(1 / R), C))
        term4 = - 0.5 * np.sum(np.sum(vx, axis=0) * np.diag(CRC))
        logZ = term1 + term2 + term3 + term4
        # print term1, term2, term3, term4

        dR2 = - 0.5 * Nb / R
        dR3 = 0.5 * np.sum((yb - Cmx)**2, axis=0) / R**2
        dR4 = 0.5 * np.diag(np.dot(C, np.dot(np.diag(np.sum(vx, axis=0)), C.T))) / R**2
        dR = dR2 + dR3 + dR4
        dR *= 2 * R

        dC3 = np.dot(np.diag(1 / R), np.einsum('na,nb->ab', yb - Cmx, mx))
        dC4 = - np.dot(np.diag(1 / R), np.dot(C, np.diag(np.sum(vx, axis=0))))
        dC = dC3 + dC4

        dmx = np.einsum('ba,na->nb', np.dot(C.T, np.diag(1 / R)), yb - Cmx)
        dvx = np.tile(np.reshape(- 0.5 * np.diag(CRC), [1, self.Din]), [Nb, 1])

        emi_grads = {'C': dC * scale, 'R': dR * scale}
        input_grads = {'mx': dmx * scale, 'vx': dvx * scale}
        return logZ * scale, input_grads, emi_grads
