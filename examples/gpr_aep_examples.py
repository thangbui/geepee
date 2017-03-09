print "importing stuff..."
import numpy as np
import pdb
import matplotlib.pylab as plt
from .context import SGPR
from scipy import special


def run_regression_1D():
	np.random.seed(42)
	N = 200
	X = np.random.rand(N,1)
	Y = np.sin(12*X) + 0.5*np.cos(25*X) + np.random.randn(N,1)*0.2
	# plt.plot(X, Y, 'kx', mew=2)

	def plot(m):
		xx = np.linspace(-0.5, 1.5, 100)[:,None]
		mean, var = m.predict_f(xx)
		zu = m.sgp_layer.zu
		mean_u, var_u = m.predict_f(zu)
		plt.figure()
		plt.plot(X, Y, 'kx', mew=2)
		plt.plot(xx, mean, 'b', lw=2)
		plt.fill_between(
			xx[:, 0], 
			mean[:, 0] - 2*np.sqrt(var[:, 0]), 
			mean[:, 0] + 2*np.sqrt(var[:, 0]), 
			color='blue', alpha=0.2)
		plt.errorbar(zu, mean_u, yerr=2*np.sqrt(var_u), fmt='ro')
		plt.xlim(-0.1, 1.1)

	# inference
	print "inference ..."
	M = 20
	model = SGPR(X, Y, M, lik='Gaussian')
	model.optimise(method='L-BFGS-B', alpha=0.01, maxiter=2000)
	plot(model)
	plt.show()

if __name__ == '__main__':
	run_regression_1D()