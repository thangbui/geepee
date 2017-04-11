import numpy as np
import scipy.linalg as spla
from scipy.spatial.distance import cdist
import weave
import pdb

# class RBF():


def compute_kernel(lls, lsf, x, z):
    ls = np.exp(lls)
    sf = np.exp(lsf)

    if x.ndim == 1:
        x = x[None, :]

    if z.ndim == 1:
        z = z[None, :]

    r2 = cdist(x, z, 'seuclidean', V=ls)**2.0
    k = sf * np.exp(-0.5 * r2)
    return k


def grad_x(lls2, lsf2, x, z):
    ls2 = np.exp(lls2)
    sf2 = np.exp(lsf2)
    if x.ndim == 1:
        x = x[None, :]
    if z.ndim == 1:
        z = z[None, :]
    r2 = cdist(x, z, 'seuclidean', V=ls2)**2.0
    k = sf2 * np.exp(-0.5 * r2)
    x_z = x[:, None, :] - z
    ls2 = np.exp(lls2)
    c = x_z / ls2
    g = k[:, :, None] * c
    return g


def compute_psi1(lls, lsf, xmean, xvar, z):
    if xmean.ndim == 1:
        xmean = xmean[None, :]

    ls = np.exp(lls)
    sf = np.exp(lsf)
    lspxvar = ls + xvar
    constterm1 = ls / lspxvar
    constterm2 = np.prod(np.sqrt(constterm1))
    # print xmean.shape, z.shape, lspxvar.shape
    r2_psi1 = cdist(xmean, z, 'seuclidean', V=lspxvar)**2.0
    psi1 = sf * constterm2 * np.exp(-0.5 * r2_psi1)
    return psi1


def compute_psi2(lls, lsf, xmean, xvar, z):
    ls = np.exp(lls)
    sf = np.exp(lsf)
    lsp2xvar = ls + 2.0 * xvar
    constterm1 = ls / lsp2xvar
    constterm2 = np.prod(np.sqrt(constterm1))

    n_psi = z.shape[0]
    v_ones_n_psi = np.ones(n_psi)
    v_ones_dim = np.ones(z.shape[1])

    D = ls
    Dnew = ls / 2.0
    Btilde = 1.0 / (Dnew + xvar)
    Vtilde = Btilde - 1.0 / Dnew
    Qtilde = 1.0 / D + 0.25 * Vtilde

    T1 = -0.5 * np.outer(np.dot((z**2) * np.outer(v_ones_n_psi,
                                                  Qtilde), v_ones_dim), v_ones_n_psi)
    T2 = +0.5 * np.outer(np.dot(z, xmean * Btilde), v_ones_n_psi)
    T3 = -0.25 * np.dot(z * np.outer(v_ones_n_psi, Vtilde), z.T)
    T4 = -0.5 * np.sum((xmean**2) * Btilde)

    M = T1 + T1.T + T2 + T2.T + T3 + T4

    psi2 = sf**2.0 * constterm2 * np.exp(M)
    return psi2


def compute_psi1_weave(lls, lsf, xmean, xvar, z):
    ls = np.exp(lls)
    sf = np.exp(lsf)
    sf = float(sf)
    M = z.shape[0]
    Q = z.shape[1]
    lspxvar = ls + xvar
    constterm1 = ls / lspxvar
    log_denom = 0.5 * np.log(constterm1)
    psi1 = np.empty((M))

    support_code = """
    #include <math.h>
    """

    code = """
    for(int m1=0; m1<M; m1++) {
        double log_psi1 = 0;
        for(int q=0; q<Q; q++) {
            double vq = xvar(q);
            double lq = ls(q);
            double z1q = z(m1, q);
            double muz = xmean(q) - z1q;
            double psi1_exp = -muz*muz/2.0/(vq+lq) + log_denom(q);
            log_psi1 += psi1_exp;
        }
        psi1(m1) = sf*exp(log_psi1);
    }
    """

    weave.inline(code, support_code=support_code,
                 arg_names=['psi1', 'M', 'Q', 'sf', 'ls',
                            'z', 'xmean', 'xvar', 'log_denom'],
                 type_converters=weave.converters.blitz)
    return psi1


def compute_psi2_weave(lls, lsf, xmean, xvar, z):
    ls = np.exp(lls)
    sf = np.exp(lsf)
    sf = float(sf)
    M = z.shape[0]
    Q = z.shape[1]
    lsp2xvar = ls + 2.0 * xvar
    constterm1 = ls / lsp2xvar
    log_denom = 0.5 * np.log(constterm1)
    psi2 = np.empty((M, M))

    support_code = """
    #include <math.h>
    """

    code = """
    for(int m1=0; m1<M; m1++) {
        for(int m2=0; m2<=m1; m2++) {
            double log_psi2 = 0;
            for(int q=0; q<Q; q++) {
                double vq = xvar(q);
                double lq = ls(q);
                double z1q = z(m1, q);
                double z2q = z(m2, q);

                double muzhat = xmean(q) - (z1q+z2q)/2.0;
                double dz = z1q-z2q;

                double psi2_exp = - dz*dz/(4.0*lq) - muzhat*muzhat/(2.0*vq+lq) + log_denom(q);
                log_psi2 += psi2_exp;
            }
            double exp_psi2 = exp(log_psi2);
            psi2(m1, m2) = sf*sf*exp_psi2;
            if (m1 != m2) {
                psi2(m2, m1) = sf*sf*exp_psi2;
            }
        }
    }
    """

    weave.inline(code, support_code=support_code,
                 arg_names=['psi2', 'M', 'Q', 'sf', 'ls',
                            'z', 'xmean', 'xvar', 'log_denom'],
                 type_converters=weave.converters.blitz)
    return psi2

# @profile


def compute_psi_weave(lls2, lsf2, xmean, xvar, z):
    ls2 = np.exp(lls2)
    sf2 = np.exp(lsf2)
    sf2 = float(sf2)
    M = z.shape[0]
    Q = z.shape[1]
    N = xmean.shape[0]
    lsp2xvar = ls2 + 2.0 * xvar
    constterm1 = ls2 / lsp2xvar
    log_denom_psi2 = 0.5 * np.log(constterm1)
    lspxvar = ls2 + xvar
    constterm2 = ls2 / lspxvar
    log_denom_psi1 = 0.5 * np.log(constterm2)
    psi2 = np.empty((N, M, M))
    psi1 = np.empty((N, M))

    support_code = """
    #include <math.h>
    """

    code = """
    for(int n=0; n<N; n++) {
        for(int m1=0; m1<M; m1++) {
            double log_psi1 = 0;
            for(int m2=0; m2<=m1; m2++) {
                double log_psi2 = 0;
                for(int q=0; q<Q; q++) {
                    double vq = xvar(n, q);
                    double lq = ls2(q);
                    double z1q = z(m1, q);
                    double z2q = z(m2, q);

                    if (m2==0) {
                        double muz = xmean(n, q) - z1q;
                        double psi1_exp = -muz*muz/2.0/(vq+lq) + log_denom_psi1(n, q);
                        log_psi1 += psi1_exp;
                    }

                    double muzhat = xmean(n, q) - (z1q+z2q)/2.0;
                    double dz = z1q-z2q;

                    double psi2_exp = - dz*dz/(4.0*lq) - muzhat*muzhat/(2.0*vq+lq) + log_denom_psi2(n, q);
                    log_psi2 += psi2_exp;
                }
                double exp_psi2 = exp(log_psi2);
                psi2(n, m1, m2) = sf2*sf2*exp_psi2;
                if (m1 != m2) {
                    psi2(n, m2, m1) = sf2*sf2*exp_psi2;
                }
            }
            psi1(n, m1) = sf2*exp(log_psi1);
        }
    }
    """

    weave.inline(code, support_code=support_code,
                 arg_names=['psi1', 'psi2', 'M', 'Q', 'N', 'sf2', 'ls2',
                            'z', 'xmean', 'xvar', 'log_denom_psi1', 'log_denom_psi2'],
                 type_converters=weave.converters.blitz)
    return psi1, psi2


def compute_psi_weave_single(lls, lsf, xmean, xvar, z):
    ls = np.exp(lls)
    sf = np.exp(lsf)
    sf = float(sf)
    M = z.shape[0]
    Q = z.shape[1]
    lsp2xvar = ls + 2.0 * xvar
    constterm1 = ls / lsp2xvar
    log_denom_psi2 = 0.5 * np.log(constterm1)
    lspxvar = ls + xvar
    constterm2 = ls / lspxvar
    log_denom_psi1 = 0.5 * np.log(constterm2)
    psi2 = np.empty((M, M))
    psi1 = np.empty((M))

    support_code = """
    #include <math.h>
    """

    code = """
    for(int m1=0; m1<M; m1++) {
        double log_psi1 = 0;
        for(int m2=0; m2<=m1; m2++) {
            double log_psi2 = 0;
            for(int q=0; q<Q; q++) {
                double vq = xvar(q);
                double lq = ls(q);
                double z1q = z(m1, q);
                double z2q = z(m2, q);

                if (m2==0) {
                    double muz = xmean(q) - z1q;
                    double psi1_exp = -muz*muz/2.0/(vq+lq) + log_denom_psi1(q);
                    log_psi1 += psi1_exp;
                }

                double muzhat = xmean(q) - (z1q+z2q)/2.0;
                double dz = z1q-z2q;

                double psi2_exp = - dz*dz/(4.0*lq) - muzhat*muzhat/(2.0*vq+lq) + log_denom_psi2(q);
                log_psi2 += psi2_exp;
            }
            double exp_psi2 = exp(log_psi2);
            psi2(m1, m2) = sf*sf*exp_psi2;
            if (m1 != m2) {
                psi2(m2, m1) = sf*sf*exp_psi2;
            }
        }
        psi1(m1) = sf*exp(log_psi1);
    }
    """

    weave.inline(code, support_code=support_code,
                 arg_names=['psi1', 'psi2', 'M', 'Q', 'sf', 'ls', 'z',
                            'xmean', 'xvar', 'log_denom_psi1', 'log_denom_psi2'],
                 type_converters=weave.converters.blitz)
    return psi1, psi2


def compute_psi_derivatives(dL_dpsi1, psi1, dL_dpsi2, psi2, ls, sf2, xmean, xvar, z):
    _dL_dvar_1, _dL_dl_1, _dL_dZ_1, _dL_dmu_1, _dL_dS_1 = \
        psi1compDer(dL_dpsi1, psi1, sf2, ls, z, xmean, xvar)
    _dL_dvar_2, _dL_dl_2, _dL_dZ_2, _dL_dmu_2, _dL_dS_2 = \
        psi2compDer(dL_dpsi2, psi2, sf2, ls, z, xmean, xvar)

    return (_dL_dvar_1 + _dL_dvar_2, _dL_dl_1 + _dL_dl_2,
            _dL_dZ_1 + _dL_dZ_2, _dL_dmu_1 + _dL_dmu_2, _dL_dS_1 + _dL_dS_2)


def compute_kfu_derivatives(dL_dkfu, kfu, ls, sf2, x, z, grad_x=False):
    return kfucompDer(dL_dkfu, kfu, sf2, ls, z, x, grad_x)


# from GPy codebase
def psi1computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi1
    # Produced intermediate results:
    # _psi1                NxM

    lengthscale2 = np.square(lengthscale)

    # psi1
    _psi1_logdenom = np.log(S / lengthscale2 + 1.).sum(axis=-1)  # N
    _psi1_log = (_psi1_logdenom[:, None] + np.einsum('nmq,nq->nm', np.square(
        mu[:, None, :] - Z[None, :, :]), 1. / (S + lengthscale2))) / (-2.)
    _psi1 = variance * np.exp(_psi1_log)

    return _psi1


def psi2computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi2
    # Produced intermediate results:
    # _psi2                MxM

    N, M, Q = mu.shape[0], Z.shape[0], mu.shape[1]
    lengthscale2 = np.square(lengthscale)

    _psi2_logdenom = np.log(2. * S / lengthscale2 +
                            1.).sum(axis=-1) / (-2.)  # N
    _psi2_exp1 = (np.square(Z[:, None, :] - Z[None, :, :]) /
                  lengthscale2).sum(axis=-1) / (-4.)  # MxM
    Z_hat = (Z[:, None, :] + Z[None, :, :]) / 2.  # MxMxQ
    denom = 1. / (2. * S + lengthscale2)
    _psi2_exp2 = -(np.square(mu) * denom).sum(axis=-1)[:, None, None] + (2 * (mu * denom).dot(
        Z_hat.reshape(M * M, Q).T) - denom.dot(np.square(Z_hat).reshape(M * M, Q).T)).reshape(N, M, M)
    _psi2 = variance * variance * \
        np.exp(_psi2_logdenom[:, None, None] +
               _psi2_exp1[None, :, :] + _psi2_exp2)
    return _psi2


def psi1compDer(dL_dpsi1, _psi1, variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi1
    # Produced intermediate results: dL_dparams w.r.t. psi1
    # _dL_dvariance     1
    # _dL_dlengthscale  Q
    # _dL_dZ            MxQ
    # _dL_dgamma        NxQ
    # _dL_dmu           NxQ
    # _dL_dS            NxQ

    lengthscale2 = np.square(lengthscale)

    Lpsi1 = dL_dpsi1 * _psi1
    Zmu = Z[None, :, :] - mu[:, None, :]  # NxMxQ
    denom = 1. / (S + lengthscale2)
    Zmu2_denom = np.square(Zmu) * denom[:, None, :]  # NxMxQ
    _dL_dvar = Lpsi1.sum() / variance
    _dL_dmu = np.einsum('nm,nmq,nq->nq', Lpsi1, Zmu, denom)
    _dL_dS = np.einsum('nm,nmq,nq->nq', Lpsi1, (Zmu2_denom - 1.), denom) / 2.
    _dL_dZ = -np.einsum('nm,nmq,nq->mq', Lpsi1, Zmu, denom)
    _dL_dl = np.einsum('nm,nmq,nq->q', Lpsi1, (Zmu2_denom +
                                               (S / lengthscale2)[:, None, :]), denom * lengthscale)

    return _dL_dvar, _dL_dl, _dL_dZ, _dL_dmu, _dL_dS


def kfucompDer(dL_dkfu, kfu, variance, lengthscale, Z, mu, grad_x):
    # here are the "statistics" for psi1
    # Produced intermediate results: dL_dparams w.r.t. psi1
    # _dL_dvariance     1
    # _dL_dlengthscale  Q
    # _dL_dZ            MxQ

    lengthscale2 = np.square(lengthscale)

    Lpsi1 = dL_dkfu * kfu
    Zmu = Z[None, :, :] - mu[:, None, :]  # NxMxQ
    _dL_dvar = Lpsi1.sum() / variance
    _dL_dZ = -np.einsum('nm,nmq->mq', Lpsi1, Zmu / lengthscale2)
    _dL_dl = np.einsum('nm,nmq->q', Lpsi1, np.square(Zmu) / lengthscale**3)
    if grad_x:
        _dL_dx = np.einsum('nm,nmq->nq', Lpsi1, Zmu / lengthscale2)
        return _dL_dvar, _dL_dl, _dL_dZ, _dL_dx
    else:
        return _dL_dvar, _dL_dl, _dL_dZ


def psi2compDer(dL_dpsi2, _psi2, variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi2
    # Produced the derivatives w.r.t. psi2:
    # _dL_dvariance      1
    # _dL_dlengthscale   Q
    # _dL_dZ             MxQ
    # _dL_dmu            NxQ
    # _dL_dS             NxQ
    N, M, Q = mu.shape[0], Z.shape[0], mu.shape[1]
    lengthscale2 = np.square(lengthscale)
    denom = 1. / (2 * S + lengthscale2)
    denom2 = np.square(denom)

    if len(dL_dpsi2.shape) == 2:
        dL_dpsi2 = (dL_dpsi2 + dL_dpsi2.T) / 2
    else:
        dL_dpsi2 = (dL_dpsi2 + np.swapaxes(dL_dpsi2, 1, 2)) / 2
    Lpsi2 = dL_dpsi2 * _psi2  # dL_dpsi2 is MxM, using broadcast to multiply N out
    Lpsi2sum = Lpsi2.reshape(N, M * M).sum(1)  # N
    tmp = Lpsi2.reshape(N * M, M).dot(Z).reshape(N, M, Q)
    Lpsi2Z = tmp.sum(1)  # NxQ
    # np.einsum('nmo,oq,oq->nq',Lpsi2,Z,Z) #NxQ
    Lpsi2Z2 = Lpsi2.reshape(N * M, M).dot(np.square(Z)).reshape(N, M, Q).sum(1)
    # np.einsum('nmo,mq,oq->nq',Lpsi2,Z,Z) #NxQ
    Lpsi2Z2p = (tmp * Z[None, :, :]).sum(1)
    Lpsi2Zhat = Lpsi2Z
    Lpsi2Zhat2 = (Lpsi2Z2 + Lpsi2Z2p) / 2

    _dL_dvar = Lpsi2sum.sum() * 2 / variance
    _dL_dmu = (-2 * denom) * (mu * Lpsi2sum[:, None] - Lpsi2Zhat)
    _dL_dS = (2 * np.square(denom)) * (np.square(mu) *
                                       Lpsi2sum[:, None] - 2 * mu * Lpsi2Zhat + Lpsi2Zhat2) - denom * Lpsi2sum[:, None]
    #     _dL_dZ = -np.einsum('nmo,oq->oq',Lpsi2,Z)/lengthscale2+np.einsum('nmo,oq->mq',Lpsi2,Z)/lengthscale2+ \
    #              2*np.einsum('nmo,nq,nq->mq',Lpsi2,mu,denom) - np.einsum('nmo,nq,mq->mq',Lpsi2,denom,Z) - np.einsum('nmo,oq,nq->mq',Lpsi2,Z,denom)
    Lpsi2_N = Lpsi2.sum(0)
    Lpsi2_M = Lpsi2.sum(2)
    _dL_dZ = -Lpsi2_N.sum(0)[:, None] * Z / lengthscale2 + Lpsi2_N.dot(Z) / lengthscale2 + \
        2 * Lpsi2_M.T.dot(mu * denom) - Lpsi2_M.T.dot(denom) * Z - (Lpsi2.reshape(N, M * M).T.dot(
            denom).reshape(M, M, Q) * Z[None, :, :]).sum(1)  # np.einsum('nmo,oq,nq->mq',Lpsi2,Z,denom)
    _dL_dl = 2 * lengthscale * ((S / lengthscale2 * denom + np.square(mu * denom)) * Lpsi2sum[:, None] + (Lpsi2Z2 - Lpsi2Z2p) / (2 * np.square(lengthscale2)) -
                                (2 * mu * denom2) * Lpsi2Zhat + denom2 * Lpsi2Zhat2).sum(axis=0)

    return _dL_dvar, _dL_dl, _dL_dZ, _dL_dmu, _dL_dS


def d_trace_MKzz_dhypers(lls, lsf, z, M, Kzz):

    dKzz_dlsf = Kzz
    ls = np.exp(lls)

    # This is extracted from the R-code of Scalable EP for GP Classification
    # by DHL and JMHL

    gr_lsf = np.sum(M * dKzz_dlsf)

    # This uses the vact that the distance is v^21^T - vv^T + 1v^2^T, where v is a vector with the l-dimension
    # of the inducing points.

    Ml = 0.5 * M * Kzz
    Xl = z * np.outer(np.ones(z.shape[0]), 1.0 / np.sqrt(ls))
    gr_lls = np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml.T, Xl**2)) + np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml, Xl**2)) \
        - 2.0 * np.dot(np.ones(Xl.shape[0]), (Xl * np.dot(Ml, Xl)))

    Xbar = z * np.outer(np.ones(z.shape[0]), 1.0 / ls)
    Mbar1 = - M.T * Kzz
    Mbar2 = - M * Kzz
    gr_z = (Xbar * np.outer(np.dot(np.ones(Mbar1.shape[ 0 ]) , Mbar1), np.ones(Xbar.shape[ 1 ])) - np.dot(Mbar1, Xbar)) +\
        (Xbar * np.outer(np.dot(np.ones(Mbar2.shape[0]), Mbar2), np.ones(
            Xbar.shape[1])) - np.dot(Mbar2, Xbar))

    # The cost of this function is dominated by five matrix multiplications with cost M^2 * D each where D is
    # the dimensionality of the data!!!

    return gr_lsf, gr_lls, gr_z
