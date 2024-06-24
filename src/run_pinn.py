from pinn import PINN
import torch
import numpy as np
from solver_params import solver_options
from data_generation import generate_traning_data

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	pinn = PINN([3, 20, 20, 20, 20, 20, 1], 2.0, device)

	traning_data = generate_traning_data(device, n_colloc=500, n_bc=100)

	pinn.train(traning_data, 15000)

	res = pinn.forward(torch.tensor([
		[.5, .5, 0.4]
	]).to(device))

	print(res)
