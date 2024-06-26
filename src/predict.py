import torch
from params import options
import numpy as np
from tqdm import tqdm
from data_generation import t_scale, domain_scale, heat_scale
import time

n = options['domain_length']

def predict(model: torch.nn.Module, iter_end: int = options["max_iter"]) -> np.ndarray:
	u = np.zeros((n, n, iter_end))
	model.eval()

	temp = np.linspace(0, n, n) / domain_scale
	X0, Y0 = np.meshgrid(temp, temp)

	X = X0.reshape([n*n, 1])
	Y = Y0.reshape([n*n, 1])

	start_time = time.time()
	for i in tqdm(range(iter_end), desc='Predicting'):

		t = (i * options['delta_t']) / t_scale
		grid_points = np.concatenate(
			(X, Y,
			np.full((n*n, 1), t)),
			axis=1)

		grid_points = torch.tensor(grid_points, dtype=torch.float32).to(model.device)

		u_pred = model.forward(grid_points)
		u_pred = u_pred.cpu().detach().numpy().reshape((n, n))

		u[:, :, i] = u_pred * heat_scale

	runtime = round(time.time() - start_time, 5)
	print(f"Time: {runtime} seconds")

	return u