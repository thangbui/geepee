import numpy as np
import scipy.linalg as spla


def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[0]))


def matrixInverse(M):
    return chol2inv(spla.cholesky(M, lower=False))


def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    i = np.argsort(evecs)[::-1]
    W = evals[:, i]
    W = W[:, :Q]
    return (X - X.mean(0)).dot(W)


def objective_wrapper(params, params_args, obj, idxs, yb, alpha):
    params_dict = unflatten_dict(params, params_args)
    f, grad_dict = obj.objective_function(
        params_dict, idxs, yb, alpha=alpha)
    g, _ = flatten_dict(grad_dict)
    g_is_fin = np.isfinite(g)
    if np.all(g_is_fin):
        return f, g
    else:
        print("Warning: inf or nan in gradient: replacing with zeros")
        return f, np.where(g_is_fin, g, 0.)


def flatten_dict(params):
    keys = params.keys()
    shapes = {}
    ind = np.zeros(len(keys), dtype=int)
    vec = np.array([])
    for i, key in enumerate(sorted(keys)):
        val = params[key]
        shapes[key] = val.shape
        val_vec = val.ravel()
        vec = np.concatenate((vec, val_vec))
        ind[i] = val_vec.shape[0]

    indices = np.cumsum(ind)[:-1]
    return vec, (keys, indices, shapes)


def unflatten_dict(params, params_args):
    keys, indices, shapes = params_args[0], params_args[1], params_args[2]
    vals = np.split(params, indices)
    params_dict = {}
    for i, key in enumerate(sorted(keys)):
        params_dict[key] = np.reshape(vals[i], shapes[key])
    return params_dict