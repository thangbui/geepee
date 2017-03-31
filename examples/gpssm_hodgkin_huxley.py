import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
from .context import aep, ep
np.random.seed(42)
import pdb

def plot_model(model, plot_title=''):
    # plot function
	mx, vx = model.get_posterior_x()
	# mins = np.min(mx, axis=0)-1
	# maxs = np.max(mx, axis=0)+1
	mins = np.min(mx, axis=0)-0.5
	maxs = np.max(mx, axis=0)+0.5
	nGrid = 50
	xspaced = np.linspace( mins[0], maxs[0], nGrid )
	yspaced = np.linspace( mins[1], maxs[1], nGrid )
	xx, yy = np.meshgrid( xspaced, yspaced )
	Xplot = np.vstack((xx.flatten(),yy.flatten())).T
	mf, vf = model.predict_f(Xplot)
	fig = plt.figure()
	plt.imshow((mf[:, 0]).reshape(*xx.shape), 
		vmin=mf.min(), vmax=mf.max(), origin='lower',
		extent=[mins[0], maxs[0], mins[1], maxs[1]], aspect='auto')
	plt.colorbar()
	plt.contour(
			xx, yy, (mf[:, 0]).reshape(*xx.shape), 
			colors='k', linewidths=2, zorder=100)
	zu = model.sgp_layer.zu
	plt.plot(zu[:, 0], zu[:, 1], 'ko', mew=0, ms=4)
	for i in range(mx.shape[0]-1):
		plt.plot(mx[i:i+2, 0], mx[i:i+2, 1], 
			'-bo', ms=3, linewidth=2, zorder=101)
	plt.xlabel(r'$x_{t-1}$')
	plt.ylabel(r'$x_{t}$')
	plt.xlim([mins[0], maxs[0]])
	plt.ylim([mins[1], maxs[1]])
	plt.title(plot_title)
	plt.savefig('/tmp/gpssm_hh_dim_0.pdf')

	if mx.shape[0] > 1:
		fig = plt.figure()
		plt.imshow((mf[:, 1]).reshape(*xx.shape), 
			vmin=mf.min(), vmax=mf.max(), origin='lower',
			extent=[mins[0], maxs[0], mins[1], maxs[1]], aspect='auto')
		plt.colorbar()
		plt.contour(
				xx, yy, (mf[:, 1]).reshape(*xx.shape), 
				colors='k', linewidths=2, zorder=100)
		zu = model.sgp_layer.zu
		plt.plot(zu[:, 0], zu[:, 1], 'ko', mew=0, ms=4)
		for i in range(mx.shape[0]-1):
			plt.plot(mx[i:i+2, 0], mx[i:i+2, 1], 
				'-bo', ms=3, linewidth=2, zorder=101)
		plt.xlabel(r'$x_{t-1}$')
		plt.ylabel(r'$x_{t}$')
		plt.xlim([mins[0], maxs[0]])
		plt.ylim([mins[1], maxs[1]])
		plt.title(plot_title)
		plt.savefig('/tmp/gpssm_hh_dim_1.pdf')

def model_V_n_old_working():
	# load dataset
	data = np.loadtxt('./sandbox/hh_data.txt')
	# use the voltage and potasisum current
	y = data[:, [0, 2]]
	y = y / np.std(y, axis=0)
	# init hypers
	Dlatent = 2
	Dobs = y.shape[1]
	M = 40
	T = y.shape[0]
	C = np.ones([Dobs, Dlatent]) / (Dobs*Dlatent)
	R = np.ones(Dobs)*np.log(0.01)/2
	lls = np.log(3) * np.ones([Dlatent, ])
	lsf = np.reshape(np.log(1), [1, ])
	zu = y[np.random.randint(0, T, size=M), 0:Dlatent]
	lsn = np.log(0.01)/2
	params = {'ls': lls, 'sf': lsf, 'sn': lsn, 'R': R, 'C': C, 'zu': zu}

	alpha = 0.4
	print 'alpha = %.3f' % alpha
	# create AEP model
	model_aep = aep.SGPSSM(y, Dlatent, M, 
	    lik='Gaussian', prior_mean=0, prior_var=1000)
	hypers = model_aep.init_hypers(y)
	for key in params.keys():
	    hypers[key] = params[key]
	model_aep.update_hypers(hypers, alpha)
	# optimise
	model_aep.set_fixed_params(['R', 'sn', 'sf'])
	model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=10000, reinit_hypers=False)
	opt_hypers = model_aep.get_hypers()
	plot_model(model_aep, 'AEP %.3f'%alpha)
	plt.show()

def model_V_n():
	# load dataset
	data = np.loadtxt('./sandbox/hh_data.txt')
	# use the voltage and potasisum current
	y = data[:, [0, 2]]
	y = y / np.std(y, axis=0)
	# init hypers
	Dlatent = 2
	Dobs = y.shape[1]
	M = 40
	T = y.shape[0]
	R = np.ones(Dobs)*np.log(0.01)/2
	lsn = np.log(0.01)/2
	params = {'sn': lsn, 'R': R}

	alpha = 0.4
	print 'alpha = %.3f' % alpha
	# create AEP model
	model_aep = aep.SGPSSM(y, Dlatent, M, 
	    lik='Gaussian', prior_mean=0, prior_var=1000)
	hypers = model_aep.init_hypers(y)
	for key in params.keys():
	    hypers[key] = params[key]
	model_aep.update_hypers(hypers, alpha)
	# optimise
	# model_aep.set_fixed_params(['R', 'sn', 'sf'])
	model_aep.set_fixed_params(['sf'])
	model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=10000, reinit_hypers=False)
	opt_hypers = model_aep.get_hypers()
	plot_model(model_aep, 'AEP %.3f'%alpha)
	plt.show()


def model_all():
	# load dataset
	data = np.loadtxt('./sandbox/hh_data.txt')
	# use the voltage and potasisum current
	y = data
	y = y / np.std(y, axis=0)
	# init hypers
	Dlatent = 2
	Dobs = y.shape[1]
	M = 30
	T = y.shape[0]
	R = np.ones(Dobs)*np.log(0.01)/2
	lsn = np.log(0.01)/2
	params = {'sn': lsn, 'R': R}

	alpha = 0.4
	print 'alpha = %.3f' % alpha
	# create AEP model
	model_aep = aep.SGPSSM(y, Dlatent, M, 
	    lik='Gaussian', prior_mean=0, prior_var=1000)
	hypers = model_aep.init_hypers(y)
	for key in params.keys():
	    hypers[key] = params[key]
	model_aep.update_hypers(hypers, alpha)
	# optimise
	# model_aep.set_fixed_params(['R', 'sn', 'sf'])
	model_aep.set_fixed_params(['sf'])
	model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=20000, reinit_hypers=False)
	# model_aep.optimise(method='adam', alpha=alpha, maxiter=10000, reinit_hypers=False, adam_lr=0.05)
	opt_hypers = model_aep.get_hypers()
	plot_model(model_aep, 'AEP %.3f'%alpha)
	plt.show()


if __name__ == '__main__':
	model_all()
	# model_V_n()
