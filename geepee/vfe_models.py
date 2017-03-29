import numpy as np
from scipy.optimize import minimize
import pdb
from scipy.cluster.vq import kmeans2

from utils import *
from kernels import *

# ideally these should be moved to some config file
jitter = 1e-6


class VI_Model(object):

    def __init__(self, y_train):
        self.y_train = y_train
        self.N = y_train.shape[0]
        self.fixed_params = []
        self.updated = False

    def init_hypers(self, x_train=None):
        pass

    def get_hypers(self):
        pass

    def optimise(
        self, method='L-BFGS-B', tol=None, reinit_hypers=True, 
        callback=None, maxiter=1000, alpha=0.5, adam_lr=0.05, **kargs):
        self.updated = False

        if reinit_hypers:
            init_params_dict = self.init_hypers()
        else:
            init_params_dict = self.get_hypers()

        init_params_vec, params_args = flatten_dict(init_params_dict)

        try:
            if method.lower() == 'adam':
                results = adam(objective_wrapper, init_params_vec,
                               step_size=adam_lr,
                               maxiter=maxiter,
                               args=(params_args, self, None, alpha))
            else:
                options = {'maxiter': maxiter, 'disp': True, 'gtol': 1e-8}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, None, alpha),
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

    def set_fixed_params(self, params):
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)


class SGPR(VI_Model):

    def __init__(self, x_train, y_train, no_pseudo):
        '''
        This only works with real-valued (Gaussian) regression

        '''

        super(SGPR, self).__init__(y_train)
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

    def objective_function_manual(self, params, alpha=1.0):
        x = self.x_train
        y = self.y_train
        N = self.N
        Dout = self.Dout
        M = self.M
        # update model with new hypers
        self.update_hypers(params)
        Kuu_noiseless = compute_kernel(2*self.ls, 2*self.sf, self.zu, self.zu)
        Kuu = Kuu_noiseless + np.diag(jitter * np.ones((M, )))
        Lu = np.linalg.cholesky(Kuu)
        sf2 = np.exp(2*self.sf)
        sn2 = np.exp(2*self.sn)
        Kuf = compute_kernel(2*self.ls, 2*self.sf, self.zu, x)
        Kfu = Kuf.T
        KuinvKuf = np.linalg.solve(Kuu, Kuf)
        Qff = np.dot(Kfu, KuinvKuf)
        Kff_diag = sf2
        Dff_diag = sf2 - np.diag(Qff)
        Kyy_bar = Qff + alpha * np.diag(Dff_diag) + sn2*np.eye(N)
        Ly = np.linalg.cholesky(Kyy_bar)
        Lyy = np.linalg.solve(Ly, y)
        term1 = 0.5 * N * np.log(2*np.pi)
        term2 = np.sum(np.log(np.diag(Ly)))
        term3 = 0.5 * np.sum(Lyy*Lyy)
        term4 = 0.5*(1-alpha)/alpha * np.sum(np.log(1 + alpha*Dff_diag/sn2))

        print term3, term2, term1, term4

        return term1 + term2 + term3 + term4

    def objective_function(self, params, idxs=None, alpha=1.0):
        x = self.x_train
        y = self.y_train
        N = self.N
        Dout = self.Dout
        M = self.M
        # update model with new hypers
        self.update_hypers(params)
        sf2 = np.exp(2*self.sf)
        sn2 = np.exp(2*self.sn)

        # compute the approximate log marginal likelihood
        Kuu_noiseless = compute_kernel(2*self.ls, 2*self.sf, self.zu, self.zu)
        Kuu = Kuu_noiseless + np.diag(jitter * np.ones((M, )))
        Lu = np.linalg.cholesky(Kuu)
        Kuf = compute_kernel(2*self.ls, 2*self.sf, self.zu, x)
        V = np.linalg.solve(Lu, Kuf)
        r = sf2 - np.sum(V*V, axis=0)
        G = alpha*r + sn2
        iG = 1.0 / G
        A = np.eye(M) + np.dot(V*iG, V.T)
        La = np.linalg.cholesky(A)
        B = np.linalg.solve(La, V)
        iGy = iG[:, None] * y
        z_tmp = np.dot(B.T, np.dot(B, iGy)) * iG[:, None]
        z = iGy - z_tmp
        term1 = 0.5 * np.sum(z*y)
        term2 = Dout * np.sum(np.log(np.diag(La)))
        term3 = 0.5 * Dout * np.sum(np.log(G))
        term4 = 0.5 * N * Dout * np.log(2*np.pi)
        c4 = 0.5 * Dout * (1-alpha) / alpha
        term5 = c4 * np.sum(np.log(1 + alpha*r/sn2))
        energy = term1 + term2 + term3 + term4 + term5
        # print term1, term2 + term3, term4, term5

        zus = self.zu / np.exp(self.ls)
        xs = x / np.exp(self.ls)
        R = np.linalg.solve(Lu.T, V)
        RiG = R*iG
        RdQ = -np.dot(np.dot(R, z), z.T) + RiG - np.dot(np.dot(RiG, B.T), B)*iG;
        dG = z**2 + (-iG + iG**2 * np.sum(B*B, axis=0)) [:, None]
        tmp = alpha*dG - (1-alpha)/(1+alpha*r[:, None]/sn2)/sn2
        RdQ2 = RdQ + R*tmp.T
        KW = Kuf*RdQ2
        KWR = Kuu_noiseless*(np.dot(RdQ2, R.T))
        P = (np.dot(KW, xs) - np.dot(KWR, zus) 
            + (np.sum(KWR, axis=1) - np.sum(KW, axis=1))[:, None]*zus)
        dzu = P / np.exp(self.ls)
        dls = (-np.sum(P*zus, axis=0) 
            - np.sum((np.dot(KW.T, zus) - np.sum(KW, axis=0)[:, None]*xs)*xs, axis=0))
        dsn = -np.sum(dG)*sn2 - (1-alpha)*np.sum(r/(1+alpha*r/sn2))/sn2;
        dsf = (np.sum(Kuf*RdQ) - alpha*np.sum(r[:, None]*dG) 
            + (1-alpha)*np.sum(r/(1+alpha*r/sn2))/sn2)

        grad_all = {'zu': dzu, 'ls': dls, 'sn': dsn, 'sf': dsf}
        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        return energy, grad_all

    def predict_f(self, inputs, alpha=1.0):
        x = self.x_train
        y = self.y_train
        N = self.N
        Dout = self.Dout
        M = self.M
        sf2 = np.exp(2*self.sf)
        sn2 = np.exp(2*self.sn)

        # compute the approximate log marginal likelihood
        Kuu_noiseless = compute_kernel(2*self.ls, 2*self.sf, self.zu, self.zu)
        Kuu = Kuu_noiseless + np.diag(jitter * np.ones((M, )))
        Lu = np.linalg.cholesky(Kuu)
        Kuf = compute_kernel(2*self.ls, 2*self.sf, self.zu, x)
        Kut = compute_kernel(2*self.ls, 2*self.sf, self.zu, inputs)
        V = np.linalg.solve(Lu, Kuf)
        r = sf2 - np.sum(V*V, axis=0)
        G = alpha*r + sn2
        iG = 1.0 / G
        A = np.eye(M) + np.dot(V*iG, V.T)
        La = np.linalg.cholesky(A)
        B = np.linalg.solve(La, V)
        yhat = y*iG[:, None]
        yhatB = np.dot(yhat.T, B.T)
        beta = np.linalg.solve(Lu.T, np.linalg.solve(La.T, yhatB.T))
        W1 = np.eye(M) - np.linalg.solve(La.T, np.linalg.solve(La, np.eye(M))).T
        W = np.linalg.solve(Lu.T, np.linalg.solve(Lu.T, W1).T)
        KtuW = np.dot(Kut.T, W)
        mf = np.dot(Kut.T, beta)
        vf = np.exp(2*self.sf) - np.sum(KtuW*Kut.T, axis=1)
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
        mf, vf = self.sgp_layer.output_probabilistic(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def init_hypers(self):
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
            ls[i] = 2*np.log(d2imed  + 1e-16)
        sf = np.log(np.array([0.5]))

        params = dict()
        params['sf'] = sf
        params['ls'] = ls
        params['zu'] = zu
        params['sn'] = np.log(0.01)
        return params

    def get_hypers(self):
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
        self.ls = params['ls']
        self.sf = params['sf']
        self.zu = params['zu']
        self.sn = params['sn']
