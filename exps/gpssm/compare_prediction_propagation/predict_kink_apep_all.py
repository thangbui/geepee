import sys
sys.path.append('../../../')
import numpy as np
import scipy.stats
import os, sys
import geepee.aep_models as aep
import pdb
np.random.seed(0)


def predict_using_train_models():    
    # alphas = [0.0001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0]
    M = 20
    alphas = [0.5]
    K = 1
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
                model_aep.load_model('../compare_energies/tmp/compare_energies_kink_model_aep_%.4f_%s.pickle' % (alpha, prop_mode))
                
                

if __name__ == '__main__':
    run_pep()
