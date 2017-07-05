import sys
sys.path.append('../../../')
sys.path.append('../../utils/')
import geepee.vfe_models as vfe
from geepee.utils import unflatten_dict
import math
import numpy as np
from metrics import *
import time
import argparse
import os

parser = argparse.ArgumentParser(description='run binary classification experiment',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset',
            action="store", dest="dataset",
            help="dataset name, eg. boston, power", default="australian")
parser.add_argument('-m', '--pseudos', type=int,
            action="store", dest="n_pseudos",
            help="number of pseudo points, eg. 10", default=10)
parser.add_argument('-b', '--minibatch', type=int,
            action="store", dest="minibch_size",
            help="minibatch size, eg. 10", default=50)
parser.add_argument('-e', '--no_epochs', type=int,
            action="store", dest="no_epochs",
            help="number of epochs, eg. 100", default=100)
parser.add_argument('-s', '--seed', type=int,
            action="store", dest="random_seed",
            help="random seed, eg. 10", default=123)
parser.add_argument('-l', '--lrate', type=float,
            action="store", dest="lrate",
            help="adam learning rate", default=0.0001)
parser.add_argument('-nx', '--normx',
            action="store_true", dest="normalise_x",
            help="normalise_x", default=True)

args = parser.parse_args()

name = args.dataset
M = args.n_pseudos
no_epochs = args.no_epochs
no_points_per_mb = args.minibch_size
random_seed = args.random_seed
np.random.seed(random_seed)
lrate = args.lrate
normalise_x = args.normalise_x
compute_test = False

fnames = {'australian': 'australian',
          'breast': 'breast',
          'crabs': 'crabs',
          'iono': 'iono',
          'pima': 'pima',
          'sonar': 'sonar'}

# We load the dataset
datapath = '/scratch/tdb40/datasets/bincla/' + fnames[name] + '/'
datafile = datapath + 'data.txt'
data = np.loadtxt(datafile)

# We obtain the features and the targets
xindexfile = datapath + 'index_features.txt'
yindexfile = datapath + 'index_target.txt'
xindices = np.loadtxt(xindexfile, dtype=np.int32)
yindex = np.loadtxt(yindexfile, dtype=np.int32)
X = data[:, xindices]
y = data[:, yindex]
y[np.where(y == 0)] = -1
y[np.where(y == 2)] = -1
y = y.reshape([y.shape[0], 1])


# We obtain the number of splits available
nosplits_file = datapath + 'n_splits.txt'
nosplits = np.loadtxt(nosplits_file, dtype=np.int8)

# prepare output files
output_path = '/scratch/tdb40/geepee_gpc_vfe_results/'
test_iters = np.array([100, 500, 1000, 2000, 3000])
outfile1s, outfile2s = [], []
for i in test_iters:
    outname1 = output_path + name + '_vfe_' + str(M) + '_iter_' + str(i) +'.error'
    if not os.path.exists(os.path.dirname(outname1)):
        os.makedirs(os.path.dirname(outname1))
    outfile1 = open(outname1, 'w')
    outname2 = output_path + name + '_vfe_' + str(M) + '_iter_' + str(i) + '.nll'
    outfile2 = open(outname2, 'w')
    outfile1s.append(outfile1)
    outfile2s.append(outfile2)
outname3 = output_path + name + '_vfe_' + str(M) + '.time'
outfile3 = open(outname3, 'w')

for i in range(nosplits):
    print 'running split', i

    train_ind_file = datapath + 'index_train_' + str(i) + '.txt'
    test_ind_file = datapath + 'index_test_' + str(i) + '.txt'
    index_train = np.loadtxt(train_ind_file, dtype=np.int32)
    index_test = np.loadtxt(test_ind_file, dtype=np.int32)
    X_train = X[index_train, :]
    y_train = y[index_train, :]
    X_test = X[index_test, :]
    y_test = y[index_test, :]
    y_test = y_test[:, 0]

    if normalise_x:
        # we normalise the train input
        std_X_train = np.std(X_train, 0)
        std_X_train[ std_X_train == 0 ] = 1
        mean_X_train = np.mean(X_train, 0)
        X_train = (X_train - mean_X_train) / std_X_train
        X_test = (X_test - mean_X_train) / std_X_train

    # We create the sparse gp object
    model = vfe.SGPR(X_train, y_train, M, lik='probit')

    def callback(params, iteration, args):
        global X_test
        global y_test
        if (iteration + 1) in test_iters:
            idx = np.where(test_iters == (iteration + 1))[0][0]
            outfile1 = outfile1s[idx]
            outfile2 = outfile2s[idx]
            params_dict = unflatten_dict(params, args[0])
            model.update_hypers(params_dict)
            # We make predictions for the test set
            mf, vf = model.predict_f(X_test)
            mf, vf = mf[:, 0], vf[:, 0]
            # We compute the test error and log lik
            test_nll = compute_nll(y_test, mf, vf, 'cdf')
            outfile2.write('%.6f\n' % test_nll)
            outfile2.flush()
            os.fsync(outfile2.fileno())

            test_error = compute_error(y_test, mf, vf, 'cdf')
            outfile1.write('%.6f\n' % test_error)
            outfile1.flush()
            os.fsync(outfile1.fileno())
    
    # train
    t0 = time.time()
    final_params, energies = model.optimise(
        method='adam', 
        mb_size=no_points_per_mb, 
        adam_lr=lrate,
        maxiter=no_epochs,
        return_cost=True,
        callback=callback)
    t1 = time.time()

    outfile3.write('%.6f\n' % (t1-t0))
    outfile3.flush()
    os.fsync(outfile3.fileno())

    outname4 = output_path + name + '_vfe_' + str(M) + '_' + str(i) + '.energy'
    np.savetxt(outname4, energies, '%.5f')

    print 'finishing split', i

for i, idx in enumerate(test_iters):
    outfile1s[i].close()
    outfile2s[i].close()
outfile3.close()