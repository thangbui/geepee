import numpy as np
import scipy.linalg as npalg
import scipy.stats as stats
import math
import pdb


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def make_batches(N_data, batch_size):
    return [slice(i, min(i + batch_size, N_data)) for i in range(0, N_data, batch_size)]


def compute_test_error(y, m, lik, median=False):
    if lik == 'Gaussian':
        if median:
            rmse = np.sqrt(np.median((y - m)**2))
        else:
            rmse = np.sqrt(np.mean((y - m)**2))
        return rmse
    elif lik == 'Probit':
        y = y.reshape((y.shape[0],))
        y_predicted = -1 * np.ones(y.shape[0])
        y_predicted[m[:, 0] > 0] = 1
        diff = y * 1.0 * y_predicted
        return list(diff).count(-1) * 1.0 / y.shape[0]
    elif lik == 'Softmax':
        y_labels = np.argmax(y, axis=1)
        predicted_labels = np.argmax(m, axis=1)
        diff = y_labels == predicted_labels
        # keyboard()
        return 1 - sum(diff) * 1.0 / y.shape[0]


def compute_test_error_tmp(y, m, v, lik, median=False, n_samples=1000):
    if lik == 'Gaussian':
        if median:
            rmse = np.sqrt(np.median((y - m)**2))
        else:
            rmse = np.sqrt(np.mean((y - m)**2))
        return rmse
    elif lik == 'Probit':
        N_test = m.shape[0]
        epsilon = np.random.normal(0, 1, (n_samples, N_test))
        # epsilon = np.expand_dims(epsilon, 1
        m = np.reshape(m, (1, N_test))
        v = np.reshape(v, (1, N_test))

        fdrawn = np.sqrt(v) * epsilon + m
        predicted_labels = fdrawn > 0
        y_labels = np.reshape(y, (1, N_test))
        diff = y_labels == predicted_labels
        error = 1 - np.sum(diff) * 1.0 / N_test / n_samples

        y = y.reshape((y.shape[0],))
        y_predicted = -1 * np.ones(y.shape[0])
        y_predicted[m[:, 0] > 0] = 1
        diff = y * 1.0 * y_predicted
        return list(diff).count(-1) * 1.0 / y.shape[0]
    elif lik == 'Softmax':
        idxs = make_batches(m.shape[0], 200)
        py = np.zeros((m.shape[0],))
        error_sum = 0
        for idx in idxs:
            mfi = m[idx, :]
            vfi = v[idx, :]
            yi = y[idx, :]
            n_classes = mfi.shape[1]
            N_test = mfi.shape[0]
            epsilon = np.random.normal(0, 1, (n_samples, N_test, n_classes))
            # epsilon = np.expand_dims(epsilon, 1
            mi = np.expand_dims(mfi, 0)
            vi = np.expand_dims(vfi, 0)
            fdrawn = np.sqrt(vi) * epsilon + mi
            predicted_labels = np.argmax(fdrawn, axis=2)
            y_labels = np.argmax(yi, axis=1)
            y_labels = np.expand_dims(y_labels, axis=0)
            diff = y_labels != predicted_labels
            error_sum += np.sum(diff)

        return error_sum * 1.0 / N_test / n_samples


def compute_test_nll(y, mf, vf, lik, median=False, n_samples=1000):
    if lik == 'Gaussian':
        ll = -0.5 * np.log(2 * math.pi * vf) - 0.5 * (y - mf)**2 / vf
        nll = -ll
        if median:
            return np.median(nll)
        else:
            return np.mean(nll)
    elif lik == 'Probit':
        y = y.reshape((y.shape[0], 1))
        nll = - stats.norm.logcdf(1.0 * y * mf / np.sqrt(1 + vf))
        if median:
            return np.median(nll)
        else:
            return np.mean(nll)
    elif lik == 'Softmax':
        idxs = make_batches(mf.shape[0], 200)
        py = np.zeros((mf.shape[0],))
        for idx in idxs:
            mfi = mf[idx, :]
            vfi = vf[idx, :]
            yi = y[idx, :]
            n_classes = mfi.shape[1]
            N_test = mfi.shape[0]
            epsilon = np.random.normal(0, 1, (n_samples, N_test, n_classes))
            m = np.expand_dims(mfi, 0)
            v = np.expand_dims(vfi, 0)
            w = np.sqrt(v) * epsilon + m
            maxes = np.amax(w, axis=2)
            maxes = maxes.reshape(maxes.shape[0], maxes.shape[1], 1)
            exp_w = np.exp(w - maxes)
            exp_w_sum = np.sum(exp_w, axis=2).reshape((n_samples, N_test, 1))
            exp_w_sum_mat = np.tile(exp_w_sum, (1, n_classes))
            exp_w_div = exp_w / exp_w_sum_mat
            k, j = np.meshgrid(np.arange(N_test), np.arange(n_samples))
            exp_w_y = exp_w_div[j, k, np.argmax(yi, axis=1)]
            p_y = 1.0 / n_samples * np.sum(exp_w_y, axis=0)
            py[idx] = p_y

        return -np.mean(np.log(py))
        # return -10
        # raise NotImplementedError('TODO')


def compute_test_nll_tmp(y, mf, vf, lik, median=False, n_samples=1000):
    if lik == 'Gaussian':
        ll = -0.5 * np.log(2 * math.pi * vf) - 0.5 * (y - mf)**2 / vf
        nll = -ll
        if median:
            return np.median(nll)
        else:
            return np.mean(nll)
    elif lik == 'Probit':
        y = y.reshape((y.shape[0], 1))
        nll = - stats.norm.logcdf(1.0 * y * mf / np.sqrt(1 + vf))
        if median:
            return np.median(nll)
        else:
            return np.mean(nll)
    elif lik == 'Softmax':
        idxs = make_batches(mf.shape[0], 200)
        py = np.zeros(mf.shape[0])
        for idx in idxs:
            mfi = mf[idx, :]
            vfi = vf[idx, :]
            yi = y[idx, :]
            n_classes = mfi.shape[1]
            N_test = mfi.shape[0]
            epsilon = np.random.normal(0, 1, (n_samples, N_test, n_classes))
            m = np.expand_dims(mfi, 0)
            v = np.expand_dims(vfi, 0)
            w = np.sqrt(v) * epsilon + m
            maxes = np.amax(w, axis=2)
            maxes = maxes.reshape(maxes.shape[0], maxes.shape[1], 1)
            exp_w = np.exp(w - maxes)
            exp_w_sum = np.sum(exp_w, axis=2).reshape((n_samples, N_test, 1))
            exp_w_sum_mat = np.tile(exp_w_sum, (1, n_classes))
            exp_w_div = exp_w / exp_w_sum_mat
            exp_w_y = exp_w_div[:, :, np.argmax(yi, axis=1)]
            p_y = 1.0 / n_samples * np.sum(exp_w_y, axis=0)
            py[idx] = p_y

        return np.mean(np.log(py))
        # raise NotImplementedError('TODO')


def is_positive_definite(x):
    try:
        U = np.linalg.cholesky(x)
        return True
    except np.linalg.linalg.LinAlgError as e:
        print(e)
        return False
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

def is_any_nan(x):
    return np.any(np.isnan(x))

def compute_distance_matrix(x):
    diff = (x[:, None, :] - x[None, :, :])
    dist = abs(diff)
    return dist

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
    for i in range(len(y_hat)):
        if y_actual[i] == 1 and y_actual != y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
    for i in range(len(y_hat)):
        if y_actual[i] == 0 and y_actual != y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)
