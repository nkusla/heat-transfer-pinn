import numpy as np

options = {
	'alpha' : 2.0,
	'delta_x' : 1,
	'delta_t' : 0.125,
	'domain_length' : 100,
	'max_iter' : 600,
	't_end' : 0.0
}

options['t_end'] = options['max_iter'] * options['delta_t']

def boundaries(u: np.ndarray, k: float, delta_t: float):
	len = u.shape[0]

	initial_temp = 100.0

	u[len-1:,:,k] = np.full(u.shape[0], initial_temp)
	u[:,len-1,k] = np.full(u.shape[0], 0.0)
	u[0,:,k] = np.full(u.shape[0], 0.0)
	u[:,0,k] = np.full(u.shape[0], 0.0)

	# u_center(u)

def u_center(u: np.ndarray):
	mid = u.shape[0] // 2
	len = round(u.shape[0] * 0.2)

	u[mid-len:mid+len:,
   		mid-len:mid+len:] = 100

def u_inital():
	length = round(options['domain_length'] / options['delta_x'])
	return np.zeros((length, length))

	# u = np.zeros((100, 100))
	# u_center(u)
	# return u

u0 = u_inital()