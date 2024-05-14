import numpy as np

options = {
	'alpha' : 1.0,
	'delta_x' : 1.0,
	'delta_t' : 0.05,
	'domain_length' : 100,
	'max_iter' : 250
}

def u_up(u, k):
	return np.full(u.shape[0], 100)

def u_right(u, k):
	return np.full(u.shape[0], 0)

def u_down(u, k):
	return np.full(u.shape[0], 0)

def u_left(u, k):
	return np.full(u.shape[0], 0)

def u_inital():
	length = round(options['domain_length'] / options['delta_x'])
	return np.zeros((length, length))

boundary = (u_up, u_right, u_down, u_left)
u0 = u_inital()