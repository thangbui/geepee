import sys
sys.path.append('../../../')
import numpy as np
import scipy.stats
import os, sys
import geepee.aep_models as aep
from geepee.config import PROP_MC, PROP_MM
import pdb
from scipy.misc import logsumexp
np.random.seed(0)


def predict_using_trained_models():    
    # alphas = [0.0001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0]
    M = 20
    alphas = [0.5]
    K = 1
    T_test = 20
    mm_se = np.zeros((K, T_test, len(alphas)))
    mm_ll = np.zeros((K, T_test, len(alphas)))
    mc_se = np.zeros((K, T_test, len(alphas)))
    mc_ll = np.zeros((K, T_test, len(alphas)))
    for k in range(K):
        y_train = np.loadtxt('data/kink_train_%d.txt'%k)
        y_train = np.reshape(y_train, [y_train.shape[0], 1])
        y_test = np.loadtxt('data/kink_test_%d.txt'%k)
        y_test = np.reshape(y_test, [y_test.shape[0], 1])
        
        for i, alpha in enumerate(alphas):
            # for prop_mode in [PROP_MM, PROP_MC]:
            for prop_mode in [PROP_MM]:
                print 'k %d, alpha = %.6f, prop_mode %s' % (k, alpha, prop_mode)
                model_aep = aep.SGPSSM(y_train, 1, M, 
                    lik='Gaussian', prior_mean=0, prior_var=1)
                model_aep.load_model('trained_models/kink_aep_model_index_%d_M_%d_alpha_%.4f_%s.pickle' % (k, M, alpha, prop_mode))
                
                # predict using MM and MC, TODO: lin
                _, _, my_MM, _, vy_MM = model_aep.predict_forward(T_test, prop_mode=PROP_MM)
                _, my_MC, vy_MC = model_aep.predict_forward(T_test, prop_mode=PROP_MC, no_samples=500)
                
                y_test = y_test[:, 0]

                my_MM = my_MM[:, 0]
                vy_MM = vy_MM[:, 0]

                my_MC = my_MC[:, :, 0].T
                vy_MC = vy_MC[:, :, 0].T

                mm_se[k, :, i] = (my_MM - y_test)**2
                mm_ll[k, :, i] = -0.5 * np.log(2*np.pi*vy_MM) - 0.5*(my_MM-y_test)**2/vy_MM

                mc_se[k, :, i] = (np.mean(my_MC, axis=0) - y_test)**2
                mc_ll[k, :, i] = logsumexp(-0.5*np.log(2*np.pi*vy_MC) - 0.5*(my_MC-y_test)**2/vy_MC, axis=0) - np.log(my_MC.shape[0])             

    mm_se_mean = np.mean(mm_se, axis=0)
    mm_se_error = np.std(mm_se, axis=0) / np.sqrt(K)
    mm_ll_mean = np.mean(mm_ll, axis=0)
    mm_ll_error = np.std(mm_ll, axis=0) / np.sqrt(K)
    mc_se_mean = np.mean(mc_se, axis=0)
    mc_se_error = np.std(mc_se, axis=0) / np.sqrt(K)
    mc_ll_mean = np.mean(mc_ll, axis=0)
    mc_ll_error = np.std(mc_ll, axis=0) / np.sqrt(K)
    
    np.savetxt('res/kink_mm_se_mean.txt', mm_se_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mm_se_error.txt', mm_se_error, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mm_ll_mean.txt', mm_ll_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mm_ll_error.txt', mm_ll_error, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_se_mean.txt', mc_se_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_se_error.txt', mc_se_error, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_ll_mean.txt', mc_ll_mean, fmt='%.5f', delimiter=',')
    np.savetxt('res/kink_mc_ll_error.txt', mc_ll_error, fmt='%.5f', delimiter=',')           

if __name__ == '__main__':
    predict_using_trained_models()
