import numpy as np


def step(x):
	y = x.copy()
	y[y < 0.0] = 0.0
	y[y > 0.0] = 1.0
	return y + 0.05*np.random.randn(x.shape[0], 1)

def spiral(no_samples, noise=0.0):
	def genSpiral(N, delta):
		X = np.zeros((N, 2))
		for i in range(N):
			r = i * 5.0 / N
			t = 1.75 * i / N * 2.0 * np.pi + delta
			x = r * np.sin(t) + (2*np.random.rand() - 1) * noise
			y = r * np.cos(t) + (2*np.random.rand() - 1) * noise
			X[i, :] = np.array([x, y])
		return X

	N = int(no_samples / 2.0)
	pos_x = genSpiral(N, 0)
	neg_x = genSpiral(N, np.pi)
	x = np.vstack((pos_x, neg_x))
	y = np.ones((2*N, 1))
	y[N:] = -1
	return x, y