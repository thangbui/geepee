import sys
sys.path.append('../../../')
import numpy as np
import geepee.aep_models as aep
np.random.seed(42)
import pdb


def model_gp(params_fname, M=20, alpha=0.5):
    # load dataset
    data = np.loadtxt('hh_data.txt')
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
    # optimise
    # model_aep.set_fixed_params(['sf_emission', 'sf_dynamic'])
    # model_aep.set_fixed_params(
    #     ['sf_emission', 'sf_dynamic', 'sn', 'sn_emission'])
    opt_hypers = model_aep.optimise(
        method='adam', alpha=alpha, maxiter=50000, reinit_hypers=False)
    model_aep.save_model(params_fname)


if __name__ == '__main__':
    M = 40
    alpha = 0.2
    model_gp('/tmp/hh_gpssm_M_%d_alpha_%.2f.pickle'%(M, alpha),
             M=M, alpha=alpha)
