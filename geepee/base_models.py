"""Summary
"""
import cPickle as pickle
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from config import *
from utils import ObjectiveWrapper, flatten_dict, unflatten_dict, adam
from utils import profile, PCA_reduce, matrixInverse
from lik_layers import Gauss_Layer, Probit_Layer, Gauss_Emis
from kernels import *

class Base_Model(object):
    """Summary
    
    Attributes:
        fixed_params (list): Description
        N (TYPE): Description
        updated (bool): Description
        y_train (TYPE): Description
    """

    def __init__(self, y_train):
        """Summary
        
        Args:
            y_train (TYPE): Description
        """
        self.y_train = y_train
        self.N = y_train.shape[0]
        self.fixed_params = []
        self.updated = False

    def init_hypers(self, y_train, x_train=None):
        """Summary
        
        Args:
            y_train (TYPE): Description
            x_train (None, optional): Description
        
        """
        pass

    def get_hypers(self):
        """Summary
        """
        pass

    def update_hypers(self, params):
        """Summary
        
        Args:
            params (TYPE): Description
        """
        pass

    def optimise(
            self, method='L-BFGS-B', tol=None, reinit_hypers=True,
            callback=None, maxfun=100000, maxiter=1000, alpha=0.5,
            mb_size=None, adam_lr=0.001, prop_mode=PROP_MM, disp=True, **kargs):
        """Summary
        
        Args:
            method (str, optional): Description
            tol (None, optional): Description
            reinit_hypers (bool, optional): Description
            callback (None, optional): Description
            maxfun (int, optional): Description
            maxiter (int, optional): Description
            alpha (float, optional): Description
            mb_size (None, optional): Description
            adam_lr (float, optional): Description
            prop_mode (TYPE, optional): Description
            disp (bool, optional): Description
            **kargs: Description
        
        Returns:
            TYPE: Description
        """
        self.updated = False

        if reinit_hypers:
            init_params_dict = self.init_hypers(self.y_train)
        else:
            init_params_dict = self.get_hypers()
        init_params_vec, params_args = flatten_dict(init_params_dict)
        objective_wrapper = ObjectiveWrapper()

        if mb_size is None:
            mb_size = self.N
        try:
            if method.lower() == 'adam':
                results = adam(
                    objective_wrapper, init_params_vec,
                    step_size=adam_lr,
                    maxiter=maxiter,
                    args=(params_args, self, mb_size, alpha, prop_mode),
                    disp=disp,
                    callback=callback)
                final_params = results
            else:
                options = {'maxfun': maxfun, 'maxiter': maxiter,
                           'disp': disp, 'gtol': 1e-5, 'ftol': 1e-5}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, self.N, alpha, prop_mode),
                    method=method,
                    jac=True,
                    tol=tol,
                    callback=callback,
                    options=options)
                final_params = results.x

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'
            final_params = objective_wrapper.previous_x    

        final_params = unflatten_dict(final_params, params_args)
        self.update_hypers(final_params)
        return final_params

    def set_fixed_params(self, params):
        """Summary
        
        Args:
            params (TYPE): Description
        """
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)

    def save_model(self, fname='/tmp/model.pickle'):
        """Summary
        
        Args:
            fname (str, optional): Description
        """
        params = self.get_hypers()
        pickle.dump(params, open(fname, "wb"))

    def load_model(self, fname='/tmp/model.pickle'):
        """Summary
        
        Args:
            fname (str, optional): Description
        """
        params = pickle.load(open(fname, "rb"))
        self.update_hypers(params)


class Base_SGP_Layer(object):
    """Sparse Gaussian process layer
    
    Attributes:
        A (TYPE): Description
        B_det (TYPE): Description
        B_sto (TYPE): Description
        Din (TYPE): Description
        Dout (TYPE): Description
        Kuu (TYPE): Description
        Kuuinv (TYPE): Description
        ls (TYPE): Description
        M (TYPE): Description
        mu (TYPE): Description
        N (TYPE): Description
        sf (int): Description
        Splusmm (TYPE): Description
        Su (TYPE): Description
        theta_1 (TYPE): Description
        theta_1_R (TYPE): Description
        theta_2 (TYPE): Description
        zu (TYPE): Description
    """

    def __init__(
            self, no_train, input_size, output_size, no_pseudo, 
            nat_param=True):
        """Initialisation
        
        Args:
            no_train (int): Number of training points
            input_size (int): Number of input dimensions
            output_size (int): Number of output dimensions
            no_pseudo (int): Number of pseudo-points
        """
        self.Din = Din = input_size
        self.Dout = Dout = output_size
        self.M = M = no_pseudo
        self.N = N = no_train
        self.nat_param = nat_param

        # variables for the mean and covariance of q(u)
        self.mu = np.zeros([Dout, M, ])
        self.Su = np.zeros([Dout, M, M])
        self.Suinv = np.zeros([Dout, M, M])
        self.Splusmm = np.zeros([Dout, M, M])

        # numpy variable for inducing points, Kuuinv, Kuu and its gradients
        self.zu = np.zeros([M, Din])
        self.Kuu = np.zeros([M, M])
        self.Kuuinv = np.zeros([M, M])

        # variables for the hyperparameters
        self.ls = np.zeros([Din, ])
        self.sf = 0

        # and natural parameters
        self.theta_1_R = np.zeros([Dout, M, M])
        self.theta_2 = np.zeros([Dout, M, ])
        self.theta_1 = np.zeros([Dout, M, M])

        # terms that are common to all datapoints in each minibatch
        self.A = np.zeros([Dout, M, ])
        self.B_det = np.zeros([Dout, M, M])
        self.B_sto = np.zeros([Dout, M, M])


    def forward_prop_thru_post(self, mx, vx=None, mode=PROP_MM, return_info=False):
        """Propagate input distributions through the posterior non-linearity
        
        Args:
            mx (float): means of the input distributions, size K x Din
            vx (float, optional): variances (if uncertain inputs), size K x Din
            mode (config param, optional): propagation mode (see config)
            return_info (bool, optional): Description
        
        Returns:
            specific results depend on the propagation mode provided
        
        Raises:
            NotImplementedError: Unknown propagation mode
        """
        if vx is None:
            return self._forward_prop_deterministic_thru_post(mx, return_info)
        else:
            if mode == PROP_MM:
                return self._forward_prop_random_thru_post_mm(mx, vx, return_info)
            elif mode == PROP_LIN:
                raise NotImplementedError(
                    'Prediction with linearisation not implemented TODO')
                # return self._forward_prop_random_thru_post_lin(mx, vx,
                # return_info)
            elif mode == PROP_MC:
                return self._forward_prop_random_thru_post_mc(mx, vx, return_info)
            else:
                raise NotImplementedError('unknown propagation mode')

    def _forward_prop_deterministic_thru_post(self, x, return_info=False):
        """Propagate deterministic inputs thru posterior
        
        Args:
            x (float): input values, size K x Din
            return_info (bool, optional): Description
        
        Returns:
            float, size K x Dout: output means
            float, size K x Dout: output variances
        """
        psi0 = np.exp(2 * self.sf)
        psi1 = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bpsi2 = np.einsum('dab,na,nb->nd', self.B_det, psi1, psi1)
        vout = psi0 + Bpsi2
        if return_info:
            return mout, vout, psi1
        else:
            return mout, vout

    def _forward_prop_random_thru_post_mm(self, mx, vx, return_info=False):
        """Propagate uncertain inputs thru posterior, using Moment Matching
        
        Args:
            mx (float): input means, size K x Din
            vx (TYPE): input variances, size K x Din
            return_info (bool, optional): Description
        
        Returns:
            float, size K x Dout: output means
            float, size K x Dout: output variances
        """
        psi0 = np.exp(2.0 * self.sf)
        psi1, psi2 = compute_psi_weave(
            2 * self.ls, 2 * self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bpsi2 = np.einsum('dab,nab->nd', self.B_sto, psi2)
        vout = psi0 + Bpsi2 - mout**2
        if return_info:
            return mout, vout, psi1, psi2
        else:
            return mout, vout

    def _forward_prop_random_thru_post_mc(self, mx, vx, return_info=False):
        """Propagate uncertain inputs thru posterior, using simple Monte Carlo
        
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
        m_stk, v_stk, kfu_stk = self._forward_prop_deterministic_thru_post(
            x_stk, return_info=True)
        mout = m_stk.reshape([MC_NO_SAMPLES, batch_size, self.Dout])
        vout = v_stk.reshape([MC_NO_SAMPLES, batch_size, self.Dout])
        kfu = kfu_stk.reshape([MC_NO_SAMPLES, batch_size, self.M])
        if return_info:
            return (mout, vout, kfu, x, eps), (m_stk, v_stk, kfu_stk, x_stk, e_stk)
        else:
            return mout, vout

    @profile
    def backprop_predictive_grads_lvm_mm(self, m, v, dm_dm, dm_dv, dv_dm, dv_dv,
        psi1, psi2, mx, vx):
        """Summary
        
        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm_dm (TYPE): Description
            dm_dv (TYPE): Description
            dv_dm (TYPE): Description
            dv_dv (TYPE): Description
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

        dm_all_m = dm_dm - 2 * dm_dv * m
        dpsi1 = np.einsum('nd,dm->nm', dm_all_m, self.A)
        dpsi2 = np.einsum('nd,dab->nab', dm_dv, self.B_sto)
        _, _, _, dm_dmx, dm_dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, ls, sf2, mx, vx, self.zu)

        dm_all_v = dv_dm - 2 * dv_dv * m
        dpsi1 = np.einsum('nd,dm->nm', dm_all_v, self.A)
        dpsi2 = np.einsum('nd,dab->nab', dv_dv, self.B_sto)
        _, _, _, dv_dmx, dv_dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, ls, sf2, mx, vx, self.zu)

        return dm_dmx, dm_dvx, dv_dmx, dv_dvx

    def backprop_grads_reparam(self, dx, m, v, eps):
        """Summary
        
        Args:
            dx (TYPE): Description
            m (TYPE): Description
            v (TYPE): Description
            eps (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        dx = dx.reshape(eps.shape)
        dm = np.sum(dx, axis=0)
        dv = np.sum(dx * eps, axis=0) / (2 * np.sqrt(v))
        return {'mx': dm, 'vx': dv}

    @profile
    def backprop_predictive_grads_reg(self, m, v, dm_dm, dm_dv, dv_dm, dv_dv,
        kfu, x):
        """Summary
        
        Args:
            m (TYPE): Description
            v (TYPE): Description
            dm_dm (TYPE): Description
            dm_dv (TYPE): Description
            dv_dm (TYPE): Description
            dv_dv (TYPE): Description
            kfu (TYPE): Description
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2 * self.sf)

        # compute grads wrt kfu
        dkfu_m_m = np.einsum('nd,dm->nm', dm_dm, self.A)
        dkfu_m_v = 2 * np.einsum('nd,dab,na->nb', dm_dv, self.B_det, kfu)
        dkfu_m = dkfu_m_m + dkfu_m_v
        _, _, _, dm_dx = compute_kfu_derivatives(
            dkfu_m, kfu, ls, sf2, x, self.zu, grad_x=True)

        dkfu_v_m = np.einsum('nd,dm->nm', dv_dm, self.A)
        dkfu_v_v = 2 * np.einsum('nd,dab,na->nb', dv_dv, self.B_det, kfu)
        dkfu_v = dkfu_v_m + dkfu_v_v
        _, _, _, dv_dx = compute_kfu_derivatives(
            dkfu_m, kfu, ls, sf2, x, self.zu, grad_x=True)

        return dm_dx, dv_dx

    def sample(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        Su = self.Su
        mu = self.mu
        Lu = np.linalg.cholesky(Su)
        epsilon = np.random.randn(self.Dout, self.M)
        u_sample = mu + np.einsum('dab,db->da', Lu, epsilon)

        kff = compute_kernel(2 * self.ls, 2 * self.sf, x, x)
        kff += np.diag(JITTER * np.ones(x.shape[0]))
        kfu = compute_kernel(2 * self.ls, 2 * self.sf, x, self.zu)
        qfu = np.dot(kfu, self.Kuuinv)
        mf = np.einsum('nm,dm->nd', qfu, u_sample)
        vf = kff - np.dot(qfu, kfu.T)
        Lf = np.linalg.cholesky(vf)
        epsilon = np.random.randn(x.shape[0], self.Dout)
        f_sample = mf + np.einsum('ab,bd->ad', Lf, epsilon)
        return f_sample

    def compute_kuu(self):
        """update kuu and kuuinv
        """
        ls = self.ls
        sf = self.sf
        Dout = self.Dout
        M = self.M
        zu = self.zu
        self.Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        self.Kuu += np.diag(JITTER * np.ones((M, )))
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def update_posterior(self):
        """update the posterior approximation
        """
        if self.nat_param:
            self.Suinv = self.Kuuinv + self.theta_1
            self.Su = np.linalg.inv(self.Suinv)
            # self.Su = matrixInverse(self.Suinv)
            self.mu = np.einsum('dab,db->da', self.Su, self.theta_2)
        else:
            self.Su = self.theta_1
            self.Suinv = np.linalg.inv(self.Su)
            # self.Suinv = matrixInverse(self.Su)
            self.mu = self.theta_2
        self.Splusmm = self.Su + np.einsum('da,db->dab', self.mu, self.mu)
        self.A = np.einsum('ab,db->da', self.Kuuinv, self.mu)
        self.B_sto = - self.Kuuinv + np.einsum(
            'ab,dbc->dac',
            self.Kuuinv,
            np.einsum('dab,bc->dac', self.Splusmm, self.Kuuinv))
        self.B_det = - self.Kuuinv + np.einsum(
            'ab,dbc->dac',
            self.Kuuinv,
            np.einsum('dab,bc->dac', self.Su, self.Kuuinv))

    def compute_posterior_grad_u(self, dmu, dSu):
        # return grads wrt u params and Kuuinv
        triu_ind = np.triu_indices(self.M)
        diag_ind = np.diag_indices(self.M)
        if self.nat_param:
            dSu_via_m = np.einsum('da,db->dab', dmu, self.theta_2)
            dSu += dSu_via_m
            dSuinv = - np.einsum('dab,dbc,dce->dae', self.Su, dSu, self.Su)
            dKuuinv = np.sum(dSuinv, axis=0)
            dtheta1 = dSuinv
            deta2 = np.einsum('dab,db->da', self.Su, dmu)
        else:
            deta2 = dmu
            dtheta1 = dSu
            dKuuinv = 0

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

    def init_hypers(self, x_train=None, key_suffix=''):
        """Summary
        
        Args:
            x_train (None, optional): Description
            key_suffix (str, optional): Description
        
        Returns:
            TYPE: Description
        """
        N = self.N
        M = self.M
        Din = self.Din
        Dout = self.Dout

        if x_train is None:
            ls = np.log(np.ones((Din, )) + 0.1 * np.random.rand(Din, ))
            sf = np.log(np.array([1]))
            zu = np.tile(np.linspace(-1, 1, M).reshape((M, 1)), (1, Din))
        else:
            if N < 10000:
                centroids, label = kmeans2(x_train, M, minit='points')
            else:
                randind = np.random.permutation(N)
                centroids = x_train[randind[0:M], :]
            zu = centroids

            if N < 1000:
                X1 = np.copy(x_train)
            else:
                randind = np.random.permutation(N)
                X1 = x_train[randind[:1000], :]

            x_dist = cdist(X1, X1, 'euclidean')
            triu_ind = np.triu_indices(X1.shape[0])
            ls = np.zeros((Din, ))
            d2imed = np.median(x_dist[triu_ind])
            for i in range(Din):
                ls[i] = np.log(d2imed / 2 + 1e-16)
            sf = np.log(np.array([0.5]))

        Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        Kuu += np.diag(JITTER * np.ones((M, )))
        Kuuinv = np.linalg.inv(Kuu)

        eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        eta2 = np.zeros((Dout, M))
        for d in range(Dout):
            mu = np.linspace(-1, 1, M).reshape((M, 1))
            # mu += 0.01 * np.random.randn(M, 1)
            alpha = 0.5 * np.random.rand(M)
            # alpha = 0.01 * np.ones(M)
            Su = np.diag(alpha)
            if self.nat_param:
                Suinv = np.diag(1 / alpha)
                theta2 = np.dot(Suinv, mu)
                theta1 = Suinv
            else:
                Suinv = np.diag(1 / alpha) + Kuuinv
                Su = np.linalg.inv(Suinv)
                theta1 = Su
                theta2 = np.dot(Su, mu / alpha.reshape((M, 1)))

            R = np.linalg.cholesky(theta1).T
            triu_ind = np.triu_indices(M)
            diag_ind = np.diag_indices(M)
            R[diag_ind] = np.log(R[diag_ind])
            eta1_d = R[triu_ind].reshape((M * (M + 1) / 2,))
            eta2_d = theta2.reshape((M,))
            eta1_R[d, :] = eta1_d
            eta2[d, :] = eta2_d

        params = dict()
        params['sf' + key_suffix] = sf
        params['ls' + key_suffix] = ls
        params['zu' + key_suffix] = zu
        params['eta1_R' + key_suffix] = eta1_R
        params['eta2' + key_suffix] = eta2

        return params

    def get_hypers(self, key_suffix=''):
        """Summary
        
        Args:
            key_suffix (str, optional): Description
        
        Returns:
            TYPE: Description
        """
        params = {}
        M = self.M
        Din = self.Din
        Dout = self.Dout
        params['ls' + key_suffix] = self.ls
        params['sf' + key_suffix] = self.sf
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        params_eta2 = self.theta_2
        params_eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        params_zu_i = self.zu

        for d in range(Dout):
            Rd = np.copy(self.theta_1_R[d, :, :])
            Rd[diag_ind] = np.log(Rd[diag_ind])
            params_eta1_R[d, :] = np.copy(Rd[triu_ind])

        params['zu' + key_suffix] = self.zu
        params['eta1_R' + key_suffix] = params_eta1_R
        params['eta2' + key_suffix] = params_eta2
        return params

    def update_hypers(self, params, key_suffix=''):
        """Summary
        
        Args:
            params (TYPE): Description
            key_suffix (str, optional): Description
        """
        M = self.M
        self.ls = params['ls' + key_suffix]
        self.sf = params['sf' + key_suffix]
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        zu = params['zu' + key_suffix]
        self.zu = zu

        for d in range(self.Dout):
            theta_m_d = params['eta2' + key_suffix][d, :]
            theta_R_d = params['eta1_R' + key_suffix][d, :]
            R = np.zeros((M, M))
            R[triu_ind] = np.copy(theta_R_d.reshape(theta_R_d.shape[0], ))
            R[diag_ind] = np.exp(R[diag_ind])
            self.theta_1_R[d, :, :] = R
            self.theta_1[d, :, :] = np.dot(R.T, R)
            self.theta_2[d, :] = theta_m_d

        # update Kuu given new hypers
        self.compute_kuu()
        # compute mu and Su for each layer
        self.update_posterior()


class Base_SGPLVM(Base_Model):
    """Summary
    
    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        factor_x1 (TYPE): Description
        factor_x2 (TYPE): Description
        lik_layer (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        prior_mean (TYPE): Description
        prior_var (TYPE): Description
        prior_x1 (TYPE): Description
        prior_x2 (TYPE): Description
        updated (bool): Description
        x_post_1 (TYPE): Description
        x_post_2 (TYPE): Description
    
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
        
        Raises:
            NotImplementedError: Description
        """
        super(Base_SGPLVM, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = hidden_size
        self.M = M = no_pseudo
        self.nat_param = nat_param

        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

        # natural params for latent variables
        self.factor_x1 = np.zeros((N, Din))
        self.factor_x2 = np.zeros((N, Din))

        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.prior_x1 = prior_mean / prior_var
        self.prior_x2 = 1.0 / prior_var

        self.x_post_1 = np.zeros((N, Din))
        self.x_post_2 = np.zeros((N, Din))

    def objective_function(self, params, mb_size, alpha='not_used', prop_mode=PROP_MM):
        """Summary
        
        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            alpha (str, optional): Description
            prop_mode (TYPE, optional): Description
        """
        pass

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

    def get_posterior_x(self, idxs=None):
        """Summary
        
        Returns:
            TYPE: Description
        """
        if idxs is None:
            idxs = np.arange(self.N)
        vx = 1.0 / self.x_post_2[idxs, :]
        mx = self.x_post_1[idxs, :] / self.x_post_2[idxs, :]
        return mx, vx

    def impute_missing(self, y, missing_mask, alpha=0.5, no_iters=10, add_noise=False):
        """Summary

        Args:
            y (TYPE): Description
            missing_mask (TYPE): Description
            alpha (float, optional): Description
            no_iters (int, optional): Description
            add_noise (bool, optional): Description

        Returns:
            TYPE: Description
        """
        # TODO
        # find latent conditioned on observed variables
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True

        N_test = y.shape[0]
        Q = self.Din
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
                self.sgp_layer.compute_logZ_and_gradients_imputation(
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

        # propagate x forward to predict missing points
        my, vy = self.sgp_layer.forward_prop_thru_post(
            post_m, post_v, add_noise=add_noise)

        return my, vy

    def init_hypers(self, y_train):
        """Summary
        
        Args:
            y_train (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # TODO: alternatitve method for non real-valued data
        post_m = PCA_reduce(y_train, self.Din)
        post_m_mean = np.mean(post_m, axis=0)
        post_m_std = np.std(post_m, axis=0) + 1e-5
        post_m = (post_m - post_m_mean) / post_m_std
        post_v = 0.1 * np.ones_like(post_m)
        x_params = {}
        if self.nat_param:
            post_2 = 1.0 / post_v
            post_1 = post_2 * post_m
            x_params['x1'] = post_1
            x_params['x2'] = np.log(post_2 - 1) / 2
        else:
            x_params['x1'] = post_m
            x_params['x2'] = np.log(post_v) / 2
        # learnt a GP mapping between hidden states
        print 'init latent function using GPR...'
        x = post_m
        y = y_train
        from vfe_models import SGPR
        reg = SGPR(x, y, self.M, 'Gaussian', self.nat_param)
        reg.set_fixed_params(['sn', 'sf', 'ls', 'zu'])
        reg.optimise(method='L-BFGS-B', maxiter=100, disp=False)
        sgp_params = reg.sgp_layer.get_hypers()
        lik_params = self.lik_layer.init_hypers()
        init_params = dict(sgp_params)
        init_params.update(lik_params)
        init_params.update(x_params)
        return init_params

    def get_hypers(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        sgp_params = self.sgp_layer.get_hypers()
        lik_params = self.lik_layer.get_hypers()
        x_params = {}
        x_params['x1'] = self.factor_x1
        x_params['x2'] = np.log(self.factor_x2) / 2.0

        params = dict(sgp_params)
        params.update(lik_params)
        params.update(x_params)
        return params

    def update_hypers(self, params):
        """Summary
        
        Args:
            params (TYPE): Description
        """
        self.sgp_layer.update_hypers(params)
        self.lik_layer.update_hypers(params)
        self.factor_x1 = params['x1']
        self.factor_x2 = np.exp(2 * params['x2'])

        if self.nat_param:
            self.x_post_1 = self.prior_x1 + self.factor_x1
            self.x_post_2 = self.prior_x2 + self.factor_x2
        else:
            self.x_post_1 = self.factor_x1 / self.factor_x2
            self.x_post_2 = 1.0 / self.factor_x2

    def compute_posterior_grad_x(self, dmx, dvx, idxs):
        grads_x_1 = np.zeros_like(self.x_post_1)
        grads_x_2 = np.zeros_like(grads_x_1)
        if self.nat_param:
            post_1 = self.x_post_1[idxs, :]
            post_2 = self.x_post_2[idxs, :]
            grads_x_1[idxs, :] = dmx / post_2
            grads_x_2[idxs, :] = -dmx * post_1 / post_2**2 - dvx / post_2**2
            grads_x_2[idxs, :] *= 2 * self.factor_x2[idxs, :]
        else:
            grads_x_1[idxs, :] = dmx
            grads_x_2[idxs, :] = dvx
            grads_x_2[idxs, :] *= 2 * self.factor_x2[idxs, :]
        grads = {}
        grads['x1'] = grads_x_1
        grads['x2'] = grads_x_2
        return grads


class Base_SGPR(Base_Model):
    """Summary
    
    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        lik_layer (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        updated (bool): Description
        x_train (TYPE): Description
    """

    def __init__(self, x_train, y_train, no_pseudo, 
        lik='Gaussian', nat_param=True):
        """Summary
        
        Args:
            x_train (TYPE): Description
            y_train (TYPE): Description
            no_pseudo (TYPE): Description
            lik (str, optional): Description
        
        Raises:
            NotImplementedError: Description
        """
        super(Base_SGPR, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = x_train.shape[1]
        self.M = M = no_pseudo
        self.x_train = x_train
        self.nat_param = nat_param

        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

    @profile
    def objective_function(self, params, mb_size, alpha=1.0, prop_mode=PROP_MM):
        """Summary
        
        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            alpha (float, optional): Description
            prop_mode (TYPE, optional): Description
        """
        pass

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


class Base_SDGPR(Base_Model):
    """Summary
    
    Attributes:
        Din (TYPE): Description
        Dout (TYPE): Description
        L (TYPE): Description
        lik_layer (TYPE): Description
        Ms (TYPE): Description
        N (TYPE): Description
        size (TYPE): Description
        updated (bool): Description
        x_train (TYPE): Description
    """

    def __init__(self, x_train, y_train, no_pseudos, hidden_sizes, 
        lik='Gaussian'):
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
        super(Base_SDGPR, self).__init__(y_train)
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
        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

    def objective_function(self, params, mb_size, alpha=1.0, prop_mode=PROP_MM):
        """Summary
        
        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            alpha (float, optional): Description
            prop_mode (TYPE, optional): Description
        """
        pass

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

    def predict_f_with_input_grad(self, inputs):
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

        # propagate inputs forward
        mout, vout, psi1, psi2 = [], [], [], []
        for i, layer in enumerate(self.sgp_layers):
            if i == 0:
                # first layer
                m0, v0, kfu0 = layer.forward_prop_thru_post(
                    inputs, return_info=True)
                mout.append(m0)
                vout.append(v0)
                psi1.append(kfu0)
                psi2.append(None)
            else:
                mi, vi, psi1i, psi2i = layer.forward_prop_thru_post(
                    mout[i - 1], vout[i - 1], return_info=True)
                mout.append(mi)
                vout.append(vi)
                psi1.append(psi1i)
                psi2.append(psi2i)

        dm_dm = np.ones((1, 1))
        dm_dv = np.zeros((1, 1))
        dv_dm = np.zeros((1, 1))
        dv_dv = np.ones((1, 1))
        grad_list = []
        for i in range(self.L - 1, -1, -1):
            layer = self.sgp_layers[i]
            if i == 0:
                grad_input = layer.backprop_predictive_grads_reg(
                    mout[i], vout[i], dm_dm, dm_dv, dv_dm, dv_dv, psi1[i], inputs)
            else:
                grad_input = layer.backprop_predictive_grads_lvm_mm(
                    mout[i], vout[i], dm_dm, dm_dv, dv_dm, dv_dv, psi1[
                        i], psi2[i],
                    mout[i - 1], vout[i - 1])
                dm_dm, dm_dv, dv_dm, dv_dv = grad_input
        dm_dx, dv_dx = grad_input

        return mout[-1], vout[-1], dm_dx, dv_dx

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

    def predict_y_with_input_grad(self, inputs):
        """Summary
        
        Args:
            inputs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        mf, vf, dm_dx, dv_dx = self.predict_f_with_input_grad(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy, dm_dx, dv_dx

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
                sgp_params = self.sgp_layers[i].init_hypers(
                    key_suffix='_%d' % i)
            init_params.update(sgp_params)

        lik_params = self.lik_layer.init_hypers()
        init_params.update(lik_params)
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
        return params

    def update_hypers(self, params):
        """Summary
        
        Args:
            params (TYPE): Description
        """
        for i, layer in enumerate(self.sgp_layers):
            layer.update_hypers(params, key_suffix='_%d' % i)
        self.lik_layer.update_hypers(params)


class Base_SGPSSM(Base_Model):
    """Summary 
    # TODO: prediction and more robust init
    
    Attributes:
        Dcon_dyn (TYPE): Description
        Dcon_emi (TYPE): Description
        Din (TYPE): Description
        Dout (TYPE): Description
        emi_layer (TYPE): Description
        gp_emi (TYPE): Description
        lik_layer (TYPE): Description
        M (TYPE): Description
        N (TYPE): Description
        sn (TYPE): Description
        updated (bool): Description
        x_control (TYPE): Description
        x_factor_1 (TYPE): Description
        x_factor_2 (TYPE): Description
        x_post_1 (TYPE): Description
        x_post_2 (TYPE): Description
        x_prior_1 (TYPE): Description
        x_prior_2 (TYPE): Description
    
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
        
        Raises:
            NotImplementedError: Description
        """
        super(Base_SGPSSM, self).__init__(y_train)
        if x_control is not None:
            self.Dcon_dyn = Dcon_dyn = x_control.shape[1]
            self.x_control = x_control
            if control_to_emi:
                self.Dcon_emi = Dcon_emi = x_control.shape[1]
            else:
                self.Dcon_emi = Dcon_emi = 0
        else:
            self.Dcon_dyn = Dcon_dyn = 0
            self.Dcon_emi = Dcon_emi = 0
            self.x_control = None
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = hidden_size
        self.M = M = no_pseudo
        self.gp_emi = gp_emi
        self.nat_param = nat_param

        if gp_emi:
            if lik.lower() == 'gaussian':
                self.lik_layer = Gauss_Layer(N, Dout)
            elif lik.lower() == 'probit':
                self.lik_layer = Probit_Layer(N, Dout)
            else:
                raise NotImplementedError('likelihood not implemented')
        else:
            if lik.lower() == 'gaussian':
                self.emi_layer = Gauss_Emis(y_train, Dout, Din + Dcon_emi)
            elif lik.lower() == 'probit':
                self.emi_layer = Probit_Emis(y_train, Dout, Din + Dcon_emi)
            else:
                raise NotImplementedError('likelihood not implemented')

        # natural params for latent variables
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.x_factor_1 = np.zeros((N, Din))
        self.x_factor_2 = np.zeros((N, Din))
        self.x_prior_1 = prior_mean / prior_var
        self.x_prior_2 = 1.0 / prior_var

        self.x_post_1 = np.zeros((N, Din))
        self.x_post_2 = np.zeros((N, Din))

    @profile
    def objective_function(self, params, mb_size, alpha=1.0, prop_mode=PROP_MM):
        """Summary
        
        Args:
            params (TYPE): Description
            mb_size (TYPE): Description
            alpha (float, optional): Description
            prop_mode (TYPE, optional): Description
        """
        pass

    def predict_f(self, inputs):
        """Summary
        
        Args:
            inputs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        mf, vf = self.dyn_layer.forward_prop_thru_post(inputs)
        return mf, vf

    def predict_forward(self, T, x_control=None):
        """Summary
        
        Args:
            T (TYPE): Description
            x_control (None, optional): Description
        
        Returns:
            TYPE: Description
        """
        mx = np.zeros((T, self.Din))
        vx = np.zeros((T, self.Din))
        my = np.zeros((T, self.Dout))
        vy_noiseless = np.zeros((T, self.Dout))
        vy = np.zeros((T, self.Dout))
        post_m, post_v = self.get_posterior_x()
        mtm1 = post_m[[-1], :]
        vtm1 = post_v[[-1], :]
        for t in range(T):
            if self.Dcon_dyn > 0:
                mtm1 = np.hstack((mtm1, x_control[[t], :]))
                vtm1 = np.hstack((vtm1, np.zeros((1, self.Dcon_dyn))))
            mt, vt = self.dyn_layer.forward_prop_thru_post(mtm1, vtm1)
            if self.Dcon_emi > 0:
                mtc = np.hstack((mt, x_control[[t], :]))
                vtc = np.hstack((vt, np.zeros((1, self.Dcon_emi))))
            else:
                mtc, vtc = mt, vt
            if self.gp_emi:
                mft, vft = self.emi_layer.forward_prop_thru_post(mtc, vtc)
                myt, vyt_n = self.lik_layer.output_probabilistic(mft, vft)
            else:
                myt, vyt, vyt_n = self.emi_layer.output_probabilistic(mt, vt)
                vft = np.diagonal(vyt, axis1=1, axis2=2)
                vyt_n = np.diagonal(vyt_n, axis1=1, axis2=2)
            mx[t, :], vx[t, :] = mt, vt
            my[t, :], vy_noiseless[t, :], vy[t, :] = myt, vft, vyt_n
            mtm1 = mt
            vtm1 = vt
        return mx, vx, my, vy_noiseless, vy

    def predict_y(self, inputs):
        """Summary
        
        Args:
            inputs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        mf, vf = self.dyn_layer.forward_prop_thru_post(inputs)
        if self.gp_emi:
            mg, vg = self.emi_layer.forward_prop_thru_post(mf, vf)
            my, vy = self.lik_layer.output_probabilistic(mg, vg)
        else:
            my, _, vy = self.emi_layer.output_probabilistic(mf, vf)
            vy = np.diagonal(vy, axis1=1, axis2=2)
        return my, vy

    def get_posterior_x(self, idxs=None):
        """Summary
        
        Returns:
            TYPE: Description
        """
        if idxs is None:
            post_1 = self.x_post_1
            post_2 = self.x_post_2
        else:
            post_1 = self.x_post_1[idxs, :]
            post_2 = self.x_post_2[idxs, :]
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def get_posterior_y(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        mx, vx = self.get_posterior_x()
        if self.Dcon_emi > 0:
            mx = np.hstack((mx, self.x_control))
            vx = np.hstack((vx, np.zeros((self.N, self.Dcon_emi))))
        if self.gp_emi:
            mf, vf = self.emi_layer.forward_prop_thru_post(mx, vx)
            my, vyn = self.lik_layer.output_probabilistic(mf, vf)
        else:
            my, vy, vyn = self.emi_layer.output_probabilistic(mx, vx)
            vf = np.diagonal(vy, axis1=1, axis2=2)
            vyn = np.diagonal(vyn, axis1=1, axis2=2)
        return my, vf, vyn

    def init_hypers(self, y_train):
        """Summary
        
        Args:
            y_train (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # initialise latent variables and emission using a Gaussian LDS
        print 'init latent variable using LDS...'
        if self.Din == self.Dout:
            post_m = np.copy(y_train)
        else:
            from pylds.models import DefaultLDS
            model = DefaultLDS(self.Dout, self.Din, self.Dcon_dyn)
            model.add_data(y_train, inputs=self.x_control)
            # Initialize with a few iterations of Gibbs
            for _ in range(100):
                model.resample_model()
            # run EM
            for _ in range(100):
                model.EM_step()

            s = model.states_list.pop()
            s.info_E_step()
            post_m = s.smoothed_mus
            # post_v = np.diagonal(s.smoothed_sigmas, axis1=1, axis2=2)
            # scale = np.std(post_m, axis=0)
            # post_m = (post_m - np.mean(post_m, axis=0)) / scale
        post_m = post_m
        post_v = 0.1 * np.ones_like(post_m)
        ssm_params = {'sn': np.log(0.01)*np.ones(1)}
        if self.nat_param:
            post_2 = 1.0 / post_v
            post_1 = post_2 * post_m
            ssm_params['x_factor_1'] = post_1 / 3
            ssm_params['x_factor_2'] = np.log(post_2 / 3) / 2
        else:
            ssm_params['x_factor_1'] = np.copy(post_m)
            ssm_params['x_factor_2'] = np.log(post_v) / 2    
        # learn a GP mapping between hidden states
        print 'init latent function using GPR...'
        x = post_m[:self.N - 1, :]
        y = post_m[1:, :]
        if self.Dcon_dyn > 0:
            x = np.hstack((x, self.x_control[:self.N - 1, :]))
        from vfe_models import SGPR
        # from aep_models import SGPR
        reg = SGPR(x, y, self.M, 'Gaussian', self.nat_param)
        # reg.set_fixed_params(['sn', 'sf', 'ls', 'zu'])
        reg.set_fixed_params(['sn', 'sf'])
        opt_params = reg.optimise(method='L-BFGS-B', maxiter=500, disp=False)
        reg.update_hypers(opt_params)
        dyn_params = reg.sgp_layer.get_hypers(key_suffix='_dynamic')
        
        # dyn_params['ls_dynamic'] -= np.log(5)
        # dyn_params['ls_dynamic'][self.Din:] = np.log(1)

        if self.gp_emi:
            # learn a GP mapping between hidden states and observations
            print 'init emission function using GPR...'
            x = post_m
            if self.Dcon_emi > 0:
                x = np.hstack((x, self.x_control))
            y = self.y_train
            # TODO: deal with different likelihood here
            reg = SGPR(x, y, self.M, 'Gaussian', self.nat_param)
            reg.set_fixed_params(['sn', 'sf', 'ls', 'zu'])
            opt_params = reg.optimise(method='L-BFGS-B', alpha=0.5, maxiter=5000, disp=False)
            reg.update_hypers(opt_params)
            emi_params = reg.sgp_layer.get_hypers(key_suffix='_emission')
            # emi_params['ls_emission'] -= np.log(5)
            lik_params = self.lik_layer.init_hypers(key_suffix='_emission')
        else:
            emi_params = self.emi_layer.init_hypers(key_suffix='_emission')
            if isinstance(self.emi_layer, Gauss_Emis):
                if self.Din == self.Dout:
                    emi_params['C_emission'] = np.eye(self.Din)
                else:
                    emi_params['C_emission'] = s.C
        init_params = dict(dyn_params)
        init_params.update(emi_params)
        if self.gp_emi:
            init_params.update(lik_params)
        init_params.update(ssm_params)
        return init_params

    def get_hypers(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        dyn_params = self.dyn_layer.get_hypers(key_suffix='_dynamic')
        emi_params = self.emi_layer.get_hypers(key_suffix='_emission')
        ssm_params = {}
        ssm_params['x_factor_1'] = self.x_factor_1
        ssm_params['x_factor_2'] = np.log(self.x_factor_2) / 2.0
        ssm_params['sn'] = self.sn
        params = dict(dyn_params)
        params.update(emi_params)
        params.update(ssm_params)
        if self.gp_emi:
            lik_params = self.lik_layer.get_hypers(key_suffix='_emission')
            params.update(lik_params)
        return params

    def update_hypers(self, params):
        """Summary
        
        Args:
            params (TYPE): Description
        """
        self.dyn_layer.update_hypers(params, key_suffix='_dynamic')
        self.emi_layer.update_hypers(params, key_suffix='_emission')
        if self.gp_emi:
            self.lik_layer.update_hypers(params, key_suffix='_emission')
        self.sn = params['sn']
        self.x_factor_1 = params['x_factor_1']
        self.x_factor_2 = np.exp(2 * params['x_factor_2'])

        if self.nat_param:
            self.x_post_1 = 3 * self.x_factor_1
            self.x_post_2 = 3 * self.x_factor_2
            self.x_post_1[[0, -1], :] = 2 * self.x_factor_1[[0, -1], :]
            self.x_post_2[[0, -1], :] = 2 * self.x_factor_2[[0, -1], :]
            self.x_post_1[0, :] += self.x_prior_1
            self.x_post_2[0, :] += self.x_prior_2
        else:
            self.x_post_1 = self.x_factor_1 / self.x_factor_2
            self.x_post_2 = 1.0 / self.x_factor_2

    def compute_posterior_grad_x(self, dm, dv, idxs):
        grads_x_1 = np.zeros_like(self.x_post_1)
        grads_x_2 = np.zeros_like(grads_x_1)
        if self.nat_param:
            post_1 = self.x_post_1[idxs, :]
            post_2 = self.x_post_2[idxs, :]
            grad_1 = dm / post_2
            grad_2 = - dm * post_1 / post_2**2 - dv / post_2**2
            scale_x = 3.0 * np.ones((idxs.shape[0], 1))
            scale_x[np.where(idxs == 0)[0]] = 2
            scale_x[np.where(idxs == self.N-1)[0]] = 2
            grad_1 = grad_1 * scale_x
            grad_2 = grad_2 * scale_x
            grads_x_1[idxs, :] = grad_1
            grads_x_2[idxs, :] = grad_2 * 2 * self.x_factor_2[idxs, :]
        else:
            grads_x_1[idxs, :] += dm
            grads_x_2[idxs, :] += dv
            grads_x_2[idxs, :] *= 2 * self.x_factor_2[idxs, :]
        grads = {}
        grads['x_factor_1'] = grads_x_1
        grads['x_factor_2'] = grads_x_2
        return grads
