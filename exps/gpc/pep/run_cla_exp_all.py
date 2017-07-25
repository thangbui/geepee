
import os
import time
import zlib
import numpy as np
import sys
import cPickle as pickle
import json
import sys

import subprocess

processes = set()
max_processes = 10

# datasets = ['australian'] # on dirichlet
# datasets = ['breast'] # on grothendieck
# datasets = ['crabs'] # on neumann - DONE
# datasets = ['iono'] # on boole - DONE
# datasets = ['pima'] # on goedel - DONE
datasets = ['sonar'] # on dirichlet - DONE

# datasets = ['australian', 'breast', 'crabs', 'iono', 'pima','sonar']
Ms = [5, 10, 20, 50, 100]
alphas = [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]

no_epochs = 3000
mb_size = 1000
lrate = 0.001

command_list = []

for M in Ms:
	for dataset in datasets:
		for alpha in alphas:
			cmd = 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_sparse_cla.py -d ' \
				+ dataset + ' -m ' + str(M) + ' -alpha ' + str(alpha) \
				+ ' -e ' + str(no_epochs) + ' -b ' + str(mb_size) + ' -l ' + str(lrate)
			command_list.append(cmd)

for i, command in enumerate(command_list):
	print 'running', command
	processes.add(subprocess.Popen(command, shell=True))
	if len(processes) >= max_processes:
		os.wait()
		processes.difference_update([
			p for p in processes if p.poll() is not None])
		
