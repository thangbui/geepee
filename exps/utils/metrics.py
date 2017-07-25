import numpy as np
import scipy.linalg as npalg
import scipy.stats as stats
import math
from scipy.misc import logsumexp
import pdb

def draw_randn_samples(K, m, v):
    return np.random.randn(K, m.shape[0]) * np.sqrt(v) + m

def compute_error(y, m, v, lik, median=False, no_samples=50):
    if lik == 'gauss':
        y = y.reshape((y.shape[0],))
        if median:
            rmse = np.sqrt(np.median((y - m)**2))
        else:
            rmse = np.sqrt(np.mean((y - m)**2))
        return rmse
    elif lik == 'cdf':
        K = no_samples
        fs = draw_randn_samples(K, m, v).T
        log_factor = stats.norm.logcdf(np.tile(y.reshape((y.shape[0], 1)), (1, K)) * fs)
        ll = logsumexp(log_factor - np.log(K), 1)
        return 1 - np.mean(ll > np.log(0.5))

def compute_nll(y, mf, vf, lik, median=False):
    if lik == 'gauss':
        y = y.reshape((y.shape[0],))
        ll = -0.5 * np.log(2 * math.pi * vf) - 0.5 * (y - mf)**2 / vf
        nll = -ll
        if median:
            return np.median(nll)
        else:
            return np.mean(nll)
    elif lik == 'cdf':
        y = y.reshape((y.shape[0], ))
        nll = - stats.norm.logcdf(1.0 * y * mf / np.sqrt(1 + vf))
        if median:
            return np.median(nll)
        else:
            return np.mean(nll)