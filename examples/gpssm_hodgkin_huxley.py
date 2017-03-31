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
	model_aep.update_hypers(hypers)
	# optimise
	# model_aep.set_fixed_params(['R', 'sn', 'sf'])
	model_aep.set_fixed_params(['sf'])
	opt_hypers = model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=10000, reinit_hypers=False)
	plot_model(model_aep, 'AEP %.3f'%alpha)
	plt.show()
	model_aep.save_model('/tmp/gpssm_hh_VN.pickle')


def model_all():
	# load dataset
	data = np.loadtxt('./sandbox/hh_data.txt')
	# use the voltage and potasisum current
	y = data[:, :4]
	y = y / np.std(y, axis=0)
	# init hypers
	Dlatent = 2
	Dobs = y.shape[1]
	M = 40
	T = y.shape[0]
	R = np.ones(Dobs)*np.log(0.000001)/2
	lsn = np.log(0.00001)/2
	params = {'sn': lsn, 'R': R}

	alpha = 0.4
	print 'alpha = %.3f' % alpha
	# create AEP model
	model_aep = aep.SGPSSM(y, Dlatent, M, 
	    lik='Gaussian', prior_mean=0, prior_var=1000)
	hypers = model_aep.init_hypers(y)
	for key in params.keys():
	    hypers[key] = params[key]
	model_aep.update_hypers(hypers)
	# optimise
	# model_aep.set_fixed_params(['R', 'sn', 'sf'])
	# model_aep.set_fixed_params(['sf'])
	opt_hypers = model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=40000, reinit_hypers=False)
	# opt_hypers = model_aep.optimise(method='adam', alpha=alpha, maxiter=10000, reinit_hypers=False, adam_lr=0.05)
	plot_model(model_aep, 'AEP %.3f'%alpha)
	plt.show()
	model_aep.save_model('/tmp/gpssm_hh_all.pickle')


def plot_model_all():
	# load dataset
	data = np.loadtxt('./sandbox/hh_data.txt')
	# use the voltage and potasisum current
	y = data[:, :4]
	y_scales = np.std(y, axis=0)
	y = y / y_scales
	# init hypers
	Dlatent = 2
	Dobs = y.shape[1]
	M = 40
	T = y.shape[0]
	model_aep = aep.SGPSSM(y, Dlatent, M, 
	    lik='Gaussian', prior_mean=0, prior_var=1000)
	model_aep.load_model('/tmp/gpssm_hh_all.pickle')
	my, vy, vyn = model_aep.get_posterior_y()
	vy_diag = np.diagonal(vy, axis1=1, axis2=2)
	vyn_diag = np.diagonal(vyn, axis1=1, axis2=2)
	cs = ['k', 'r', 'b', 'g']
	plt.figure()
	t = np.arange(T)
	for i in range(4):
		yi = y[:, i]
		mi = my[:, i]
		vi = vy_diag[:, i]
		vin = vyn_diag[:, i]
		plt.subplot(4, 1, i+1)
		plt.fill_between(t, mi+np.sqrt(vi), mi-np.sqrt(vi), color=cs[i], alpha=0.4)
		plt.plot(t, mi, '-', color=cs[i])
		plt.plot(t, yi, '--', color=cs[i])
		plt.xlabel('t')
		plt.ylabel('y[%d]'%i)

	plt.savefig('/tmp/gpssm_hh_prediction_aep.pdf')

	# # create EP model
	# model_ep = ep.SGPSSM(y, Dlatent, M, 
 #    	lik='Gaussian', prior_mean=0, prior_var=1000)
	# # init EP model using the AEP solution
	# import cPickle as pickle
	# opt_hypers = pickle.load(open('/tmp/gpssm_hh_all.pickle', 'rb'))
	# model_ep.update_hypers(opt_hypers)
	# aep_sgp_layer = model_aep.sgp_layer
	# Nm1 = aep_sgp_layer.N
	# model_ep.sgp_layer.theta_1 = 1.0/Nm1 * np.tile(
	#     aep_sgp_layer.theta_2[np.newaxis, :, :], [Nm1, 1, 1])
	# model_ep.sgp_layer.theta_2 = 1.0/Nm1 * np.tile(
	#     aep_sgp_layer.theta_1[np.newaxis, :, :, :], [Nm1, 1, 1, 1])
	# model_ep.x_prev_1 = np.copy(model_aep.x_factor_1)
	# model_ep.x_prev_2 = np.copy(model_aep.x_factor_2)
	# model_ep.x_next_1 = np.copy(model_aep.x_factor_1)
	# model_ep.x_next_2 = np.copy(model_aep.x_factor_2)
	# model_ep.x_up_1 = np.copy(model_aep.x_factor_1)
	# model_ep.x_up_2 = np.copy(model_aep.x_factor_2)
	# model_ep.x_prev_1[0, :] = 0
	# model_ep.x_prev_2[0, :] = 0
	# model_ep.x_next_1[-1, :] = 0
	# model_ep.x_next_2[-1, :] = 0
	# # run EP
	# model_ep.inference(no_epochs=100, alpha=0.4, parallel=False, decay=0.95)

	# my, vy, vyn = model_ep.get_posterior_y()
	# vy_diag = np.diagonal(vy, axis1=1, axis2=2)
	# vyn_diag = np.diagonal(vyn, axis1=1, axis2=2)
	# cs = ['k', 'r', 'b', 'g']
	# plt.figure()
	# t = np.arange(T)
	# for i in range(4):
	# 	yi = y[:, i]
	# 	mi = my[:, i]
	# 	vi = vy_diag[:, i]
	# 	vin = vyn_diag[:, i]
	# 	plt.subplot(4, 1, i+1)
	# 	plt.fill_between(t, mi+np.sqrt(vi), mi-np.sqrt(vi), color=cs[i], alpha=0.4)
	# 	plt.plot(t, mi, '-', color=cs[i])
	# 	plt.plot(t, yi, '--', color=cs[i])
	# 	plt.xlabel('t')
	# 	plt.ylabel('y[%d]'%i)

	# plt.savefig('/tmp/gpssm_hh_prediction_ep.pdf')

def model_all_with_control():
	# TODO: predict with control
	# load dataset
	data = np.loadtxt('./sandbox/hh_data.txt')
	# use the voltage and potasisum current
	y = data[:, :4]
	xc = data[:, [-1]]
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
	model_aep.update_hypers(hypers)
	# optimise
	# model_aep.set_fixed_params(['R', 'sn', 'sf'])
	model_aep.set_fixed_params(['sf'])
	model_aep.optimise(method='L-BFGS-B', alpha=alpha, maxiter=30000, reinit_hypers=False)
	# model_aep.optimise(method='adam', alpha=alpha, maxiter=10000, reinit_hypers=False, adam_lr=0.05)
	opt_hypers = model_aep.get_hypers()
	plot_model(model_aep, 'AEP %.3f'%alpha)
	plt.show()


if __name__ == '__main__':
	# model_all()
	plot_model_all()
	# model_all_with_control()
	# model_V_n()
