
import numpy as np
import scipy.stats
import os, sys
import pdb

def lincos(T, process_noise, obs_noise, xprev=None):
    if xprev is None:
        xprev = np.random.randn()
    y = np.zeros([T, ])
    x = np.zeros([T, ])
    xtrue = np.zeros([T, ])
    for t in range(T):
        fx = -0.5*xprev + 5*np.cos(0.5*xprev)
        xtrue[t] = fx
        x[t] = fx + np.sqrt(process_noise)*np.random.randn()
        xprev = x[t]
        y[t] = x[t] + np.sqrt(obs_noise)*np.random.randn()

    return xtrue, x, y

def gen_lincos(no_splits, T_train, T_test, process_noise, obs_noise):
    for i in range(no_splits):
        np.random.seed(i)
        # generate a dataset from the lincos function above
        T = T_train + T_test
        (xtrue, x, y) = lincos(T, process_noise, obs_noise)
        y_train = y[:T_train]
        y_test = y[T_train:]
        np.savetxt('lincos_train_%d_%.2f_%.2f.txt'%(i, process_noise, obs_noise), y_train, fmt='%.4f', delimiter=' ')
        np.savetxt('lincos_test_%d_%.2f_%.2f.txt'%(i, process_noise, obs_noise), y_test, fmt='%.4f', delimiter=' ')

if __name__ == '__main__':

    # process_noise = 0.2
    # obs_noise = 0.05
    # gen_lincos(100, 200, 20, process_noise, obs_noise)

    # process_noise = 0.2
    # obs_noise = 0.2
    # gen_lincos(100, 200, 20, process_noise, obs_noise)

    # process_noise = 0.2
    # obs_noise = 1.0
    # gen_lincos(100, 200, 20, process_noise, obs_noise) 

    process_noise = 0.2
    obs_noise = 2.0
    gen_lincos(100, 200, 20, process_noise, obs_noise)   