from pinn import PINN
import torch
import numpy as np
from solver_params import solver_options

if __name__ == "__main__":
	torch.manual_seed(1616)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(torch.cuda.get_device_name())

	pinn = PINN([3, 20, 20, 20, 20, 20, 20, 20, 1], 2.0, device)

	num_colloc = 5000
	t_end = round(solver_options["delta_t"] * solver_options["max_iter"])
	x_colloc = np.random.uniform(0, solver_options["domain_length"], (num_colloc, 2))
	t_colloc = np.random.uniform(0, t_end, (num_colloc, 1))
	x_colloc = np.hstack((x_colloc, t_colloc))

	traning_data = {
		"x_data" : None,
		"y_data" : None,
		"x_bc" : None,
		"y_bc" : None,
		"x_colloc" : torch.tensor(x_colloc, dtype=torch.float32).to(device)
	}

	pinn.train(traning_data, 1000)

	