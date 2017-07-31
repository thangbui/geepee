import sys
sys.path.append('../../../')
import numpy as np
import scipy.stats
import os, sys
import geepee.aep_models as aep
from geepee.config import PROP_MC, PROP_MM
import pdb
import argparse

np.random.seed(0)


parser = argparse.ArgumentParser(description='run gpssm experiment',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', type=int,
            action="store", dest="data_index",
            help="dataset index", default=0)
parser.add_argument('-e', '--no_epochs', type=int,
            action="store", dest="no_epochs",
            help="number of epochs, eg. 100", default=20000)
parser.add_argument('-s', '--seed', type=int,
            action="store", dest="random_seed",
            help="random seed, eg. 10", default=123)
parser.add_argument('-l', '--lrate', type=float,
            action="store", dest="lrate",
            help="adam learning rate", default=0.008)
parser.add_argument('-alpha', '--alpha', type=float,
            action="store", dest="alpha",
            help="power parameter in Power EP", default=0.5)
parser.add_argument('-nx', '--dyn_noise', type=float,
            action="store", dest="dyn_noise",
            help="noise in the dynamics", default=0.2)
parser.add_argument('-ny', '--emi_noise', type=float,
            action="store", dest="emi_noise",
            help="noise in the emission", default=0.05)
parser.add_argument('-m', '--pseudos', type=int,
            action="store", dest="no_pseudos",
            help="number of pseudo points, eg. 10", default=20)

args = parser.parse_args()
index = args.data_index
no_epochs = args.no_epochs
lrate = args.lrate
alpha = args.alpha
random_seed = args.random_seed
M = args.no_pseudos
dyn_noise = args.dyn_noise
emi_noise = args.emi_noise
prop_mode = PROP_MM

print 'set %d, alpha = %.6f, prop_mode %s, dyn_noise %.2f, emi_noise %.2f' % (index, alpha, prop_mode, dyn_noise, emi_noise)

# load datasets
y_train = np.loadtxt('data/lincos_train_%d_%.2f_%.2f.txt'%(index, dyn_noise, emi_noise))
y_train = np.reshape(y_train, [y_train.shape[0], 1])
Dlatent = 1
Dobs = 1

# create AEP model
np.random.seed(42)
model_aep = aep.SGPSSM(y_train, Dlatent, M, 
    lik='Gaussian', prior_mean=0, prior_var=1)
aep_hypers = model_aep.init_hypers(y_train)
model_aep.update_hypers(aep_hypers)
# optimise
model_aep.optimise(
    method='adam', alpha=alpha, adam_lr=lrate, maxiter=no_epochs, 
    reinit_hypers=False, prop_mode=prop_mode)
model_aep.save_model('trained_models/lincos_aep_model_index_%d_%.2f_%.2f_M_%d_alpha_%.4f_%s.pickle' % (index, dyn_noise, emi_noise, M, alpha, prop_mode))
