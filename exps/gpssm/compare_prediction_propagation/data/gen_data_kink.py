
import numpy as np
import scipy.stats
import os, sys
import pdb

# We first define several utility functions

def kink(T, process_noise, obs_noise, xprev=None):
    if xprev is None:
        xprev = np.random.randn()
    y = np.zeros([T, ])
    x = np.zeros([T, ])
    xtrue = np.zeros([T, ])
    for t in range(T):
        if xprev < 4:
            fx = xprev + 1
        else:
            fx = -4*xprev + 21

        xtrue[t] = fx
        x[t] = fx + np.sqrt(process_noise)*np.random.randn()
        xprev = x[t]
        y[t] = x[t] + np.sqrt(obs_noise)*np.random.randn()

    return xtrue, x, y

def gen_kink(no_splits, T_train, T_test):
    for i in range(no_splits):
        np.random.seed(i)
        # generate a dataset from the kink function above
        T = T_train + T_test
        process_noise = 0.2
        obs_noise = 0.05
        (xtrue, x, y) = kink(T, process_noise, obs_noise)
        y_train = y[:T_train]
        y_test = y[T_train:]
        np.savetxt('kink_train_%d.txt'%i, y_train, fmt='%.4f', delimiter=' ')
        np.savetxt('kink_test_%d.txt'%i, y_test, fmt='%.4f', delimiter=' ')

if __name__ == '__main__':
    gen_kink(20, 200, 20)  