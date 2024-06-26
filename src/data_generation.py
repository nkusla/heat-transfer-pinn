import numpy as np
import torch
from scipy.stats import qmc
from params import options
from tqdm import tqdm

SEED = 1616
torch.manual_seed(SEED)
np.random.seed(SEED)
t_scale = options['t_end']
domain_scale = options['domain_length']
heat_scale = 100.0

def generate_bc(n_bc: int):
	n_per_bc = n_bc // 4
	engine = qmc.LatinHypercube(d=1, seed=SEED)
	zeros = np.zeros((n_per_bc, 1))
	ones = np.ones((n_per_bc, 1))

	left = engine.random(n_per_bc)
	left = np.concatenate((zeros, left), axis=1)
	y_left = np.full((n_per_bc, 1), 0.0)

	right = engine.random(n_per_bc)
	right = np.concatenate((ones, right), axis=1)
	y_right = np.full((n_per_bc, 1), 0.0)

	bottom = engine.random(n_per_bc)
	bottom = np.concatenate((bottom, zeros), axis=1)
	y_bottom = np.full((n_per_bc, 1), 0.0)

	top = engine.random(n_per_bc)
	top = np.concatenate((top, ones), axis=1)
	y_top = np.full((n_per_bc, 1), 100.0)

	# time
	time = np.random.uniform(0, options['t_end'], n_bc) / t_scale
	time = time.reshape(-1, 1)

	x_bc = np.concatenate((left, right, bottom, top), axis=0)
	x_bc = np.concatenate((x_bc, time), axis=1)

	y_bc = np.concatenate((y_left, y_right, y_bottom, y_top), axis=0)
	y_bc /= heat_scale

	return y_bc, x_bc

def generate_colloc(n_colloc: int):
	engine = qmc.LatinHypercube(d=2)
	x_colloc = engine.random(n_colloc)
	time = np.random.uniform(0, options['t_end'], size=(n_colloc, 1)) / t_scale
	x_colloc = np.concatenate((x_colloc, time), axis=1)

	return x_colloc

def generate_ic(n_ic: int):
	engine = qmc.LatinHypercube(d=2, seed=SEED)
	x_ic = engine.random(n_ic)
	x_ic = np.concatenate((x_ic, np.zeros((n_ic, 1))), axis=1)
	y_ic = np.zeros((n_ic, 1))

	return y_ic, x_ic

def generate_data(u: np.ndarray, n_data: int):
	delta_x = options['delta_x']
	delta_t = options['delta_t']
	domain_length = options['domain_length']

	y_data = np.empty((n_data, 1), dtype=np.float32)
	x_data = np.empty((n_data, 3), dtype=np.float32)

	rand_pos = np.random.randint(0, u.shape[1], size=(n_data, 2))
	rand_t = np.random.randint(0, u.shape[2], size=(n_data, 1))

	rand_idx = np.hstack((rand_pos, rand_t))

	for i, rand in enumerate(tqdm(rand_idx, desc="Generating experimental data")):
		y_data[i] = u[tuple(rand)]

		data = np.array([
			rand[0] * delta_x,
			rand[1] * delta_x,
			rand[2] * delta_t
		], dtype=np.float32)

		x_data[i] = data

	# Scaling the data
	x_data[:, :2] /= domain_length
	x_data[:, 2] /= t_scale
	y_data /= heat_scale

	return y_data, x_data

def generate_traning_data(
		device: torch.device,
		u: np.ndarray = None,
		n_data: int = None,
		n_colloc: int = None,
		n_bc: int = None,
		n_ic: int = None):

	x_colloc = None
	y_data, x_data = None, None
	y_ic, x_ic = None, None
	y_bc, x_bc = None, None

	if n_data is not None and u is not None:
		y_data, x_data = generate_data(u, n_data)
		x_data = torch.tensor(x_data, dtype=torch.float32, requires_grad=True).to(device)
		y_data = torch.tensor(y_data, dtype=torch.float32, requires_grad=True).to(device)

	if n_colloc is not None:
		x_colloc = generate_colloc(n_colloc)
		x_colloc = torch.tensor(x_colloc, dtype=torch.float32, requires_grad=True).to(device)

	if n_ic is not None:
		y_ic, x_ic = generate_ic(n_ic)
		x_ic = torch.tensor(x_ic, dtype=torch.float32, requires_grad=True).to(device)
		y_ic = torch.tensor(y_ic, dtype=torch.float32, requires_grad=True).to(device)

	if n_bc is not None:
		y_bc, x_bc = generate_bc(n_bc)
		x_bc = torch.tensor(x_bc, dtype=torch.float32, requires_grad=True).to(device)
		y_bc = torch.tensor(y_bc, dtype=torch.float32, requires_grad=True).to(device)

	traning_data = {
		"x_data" : x_data,
		"y_data" : y_data,
		"x_ic": x_ic,
		"y_ic": y_ic,
		"x_bc" : x_bc,
		"y_bc" : y_bc,
		"x_colloc" : x_colloc
	}

	return traning_data

def scale_alpha(alpha: float):
	return alpha * (t_scale / (domain_scale**2))

def inverse_scale_alpha(scaled_alpha: float):
	return scaled_alpha * ((domain_scale**2) / t_scale)