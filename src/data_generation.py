import numpy as np
import torch
from scipy.stats import qmc

SEED = 1616
torch.manual_seed(SEED)
t_scale = 5.0

def generate_bc(n_bc: int):
	n_per_bc = n_bc // 4
	engine = qmc.LatinHypercube(d=1, seed=SEED)
	zeros = np.zeros((n_per_bc, 1))
	ones = np.ones((n_per_bc, 1))

	left = engine.random(n_per_bc)
	left = np.concatenate((zeros, left), axis=1)

	right = engine.random(n_per_bc)
	right = np.concatenate((ones, right), axis=1)

	bottom = engine.random(n_per_bc)
	bottom = np.concatenate((bottom, zeros), axis=1)

	top = engine.random(n_per_bc)
	top = np.concatenate((top, ones), axis=1)

	# time
	time = engine.random(n_bc) * t_scale
	time = time.reshape(-1, 1)

	x_bc = np.concatenate((left, right, bottom, top), axis=0)
	x_bc = np.concatenate((x_bc, time), axis=1)

	y_bc = np.full((n_bc, 1), 100.0).reshape(-1, 1)

	return y_bc, x_bc

def generate_colloc(n_colloc: int):
	engine = qmc.LatinHypercube(d=3, seed=SEED)
	x_colloc = engine.random(n_colloc)
	x_colloc[:, 2] *= t_scale
	return x_colloc

def generate_ic(n_ic: int):
	pass

def generate_data():
	pass

def generate_traning_data(
		device: torch.device,
		n_colloc: int = None,
		n_bc: int = None,
		n_ic: int = None,
		n_data: int = None):

	if n_colloc is not None:
		x_colloc = generate_colloc(n_colloc)
	else:
		x_colloc = None

	if n_bc is not None:
		y_bc, x_bc = generate_bc(n_bc)
	else:
		y_bc, x_bc = None, None

	traning_data = {
		"x_data" : None,
		"y_data" : None,
		"x_bc" : torch.tensor(x_bc, dtype=torch.float32).to(device),
		"y_bc" : torch.tensor(y_bc, dtype=torch.float32).to(device),
		"x_colloc" : torch.tensor(x_colloc, dtype=torch.float32).to(device)
	}

	return traning_data
