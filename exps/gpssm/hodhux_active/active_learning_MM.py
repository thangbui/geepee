import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
import sys
sys.path.append('../../../')
import geepee.aep_models as aep
from geepee.config import PROP_MC, PROP_MM
np.random.seed(42)
import pdb

from hodhux_sim import HodgkinHuxley


def I_func(t, vals):
    I = 0
    for i, val in enumerate(vals):
        I += val * (t > (60*i + 10)) - val * (t > (60*i + 50))
    return I

def train_gpssm(data_fname, M, alpha, params_fname):
    data = np.loadtxt(data_fname)
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    R = np.log([0.02]) / 2
    lsn = np.log([0.02]) / 2
    params = {'sn': lsn, 'sn_emission': R}

    # create AEP model
    x_control = xc
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=x_control, gp_emi=True, control_to_emi=True)
    hypers = model_aep.init_hypers(y)
    for key in params.keys():
        hypers[key] = params[key]
    model_aep.update_hypers(hypers)
    opt_hypers = model_aep.optimise(
        method='L-BFGS-B', alpha=alpha, maxiter=50, reinit_hypers=False)
    # opt_hypers = model_aep.optimise(
    #     method='adam', alpha=alpha, maxiter=25000, reinit_hypers=False, adam_lr=0.001)
    opt_hypers = model_aep.optimise(
        method='adam', alpha=alpha, maxiter=40000, reinit_hypers=False)
    model_aep.save_model(params_fname)


def compute_predictive_entropy_given_control(data_fname, params_fname, cval, Tcontrol, M, prior=True):
    from mutual_info import entropy
    # load dataset
    data = np.loadtxt(data_fname)
    # use the voltage and potasisum current
    std_data = np.std(data, axis=0)
    data = data / std_data
    y = data[:, :4]
    xc = data[:, [-1]]
    cval = cval / std_data[-1]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = cval * np.ones([Tcontrol, 1])
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=xc, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    no_func_samples = 300
    Hy_fixed_func = np.zeros(no_func_samples)
    for k in range(no_func_samples):
        np.random.seed(k)
        _, _, my_MM, _, vy_MM = model_aep.predict_forward_fixed_function(
            Tcontrol, x_control, prop_mode=PROP_MM, starting_from_prior=prior)
        Hy_fixed_func_k = np.sum(0.5 + 0.5 * np.log(2*np.pi*vy_MM))
        Hy_fixed_func[k] = Hy_fixed_func_k

    _, _, my_MM, _, vy_MM = model_aep.predict_forward(
        Tcontrol, x_control, prop_mode=PROP_MM)
    Hy = np.sum(0.5 + 0.5 * np.log(2*np.pi*vy_MM))
    return Hy, np.mean(Hy_fixed_func), np.std(Hy_fixed_func) / np.sqrt(no_func_samples)

if __name__ == '__main__':
    no_steps = 5
    M = 30
    alpha = 0.1
    data_vals = [2, 10, 20]
    for step_id in range(no_steps):
        print 'step', step_id, '/ total', no_steps 
        np.savetxt('res/control_signals_MM_%d.txt'%step_id, np.array(data_vals), delimiter=',')

        # generating data
        t = np.arange(0.0, 60*len(data_vals), 0.5)
        I_inj = lambda x: I_func(x, data_vals)
        runner = HodgkinHuxley(t, I_inj)
        data_fname = 'res/hh_data_MM_step_%d.txt'%step_id
        runner.Main(data_fname)

        # train a gpssm based on generated data
        params_fname = 'res/hh_gpssm_MM_M_%d_alpha_%.2f_step_%d.pickle'%(M, alpha, step_id)
        train_gpssm(data_fname, M, alpha, params_fname)

        # compute predictive entropy
        Tcontrol = 50
        Nc = 101
        control_values = np.linspace(0, 25, Nc)
        from joblib import Parallel, delayed
        inputs = range(len(control_values))
        def compute_entropy(c_ind):
            print c_ind, Nc
            return compute_predictive_entropy_given_control(data_fname, params_fname, control_values[c_ind], Tcontrol, M=M)

        num_cores = 10
        results = Parallel(n_jobs=num_cores)(delayed(compute_entropy)(c_ind) for c_ind in inputs)
        Hy = np.array(results)
        np.savetxt('res/hh_gpssm_MM_entropy_%d.txt'%step_id, Hy, delimiter=',', fmt='%.4f')

        Hy = np.loadtxt('res/hh_gpssm_MM_entropy_%d.txt'%step_id, delimiter=',')
        plt.figure()
        plt.plot(control_values, Hy[:, 0], '+', color='b')
        plt.savefig('res/hh_gpssm_MM_Hy_%d.pdf'%step_id)

        plt.figure()
        plt.plot(control_values, Hy[:, 1], '*', color='r')
        plt.fill_between(control_values, Hy[:, 1] + 2*Hy[:, 2], Hy[:, 1] - 2*Hy[:, 2], 
            facecolor='r', edgecolor='r', alpha=0.4)
        plt.savefig('res/hh_gpssm_MM_Hyfixed_%d.pdf'%step_id)

        plt.figure()
        res = Hy[:, 0] - Hy[:, 1]
        res_err = np.sqrt(Hy[:, 2]**2)
        plt.plot(control_values, res, 'o', color='m')
        plt.fill_between(control_values, res + 2*res_err, res - 2*res_err, 
            facecolor='m', edgecolor='m', alpha=0.4)
        plt.savefig('res/hh_gpssm_MM_Hydiff_%d.pdf'%step_id)

        data = np.loadtxt(data_fname)
        xc = data[:, [-1]]
        plt.figure()
        plt.hist(xc, bins=26, range=[np.min(control_values), np.max(control_values)])
        plt.savefig('res/hh_gpssm_MM_Hy_xc_hist_%d.pdf'%step_id)

        max_ind = np.argmax(res)
        max_cval = control_values[max_ind]

        data_vals.append(max_cval)
