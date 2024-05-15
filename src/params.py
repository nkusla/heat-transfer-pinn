import numpy as np

options = {
	'alpha' : 1.0,
	'delta_x' : 1.0,
	'delta_t' : 0.05,
	'domain_length' : 100,
	'max_iter' : 300
}

def boundaries(u: np.ndarray, k: float, delta_t: float):
	len = u.shape[0]

	u[len-1:,:,k] = np.full(u.shape[0], 100)
	u[:,len-1,k] = np.full(u.shape[0], 0)
	u[0,:,k] = np.full(u.shape[0], 0)
	u[:,0,k] = np.full(u.shape[0], 0)

	# u_center(u)

def u_center(u: np.ndarray):
	mid = u.shape[0] // 2
	len = round(u.shape[0] * 0.2)

	u[mid-len:mid+len:,
   		mid-len:mid+len:] = 100

def u_inital():
	length = round(options['domain_length'] / options['delta_x'])
	return np.zeros((length, length))

u0 = u_inital()