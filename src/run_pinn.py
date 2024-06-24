from pinn import PINN
import torch
import numpy as np
from solver_params import solver_options
from data_generation import generate_traning_data
from plotter import animate_plot, plot_frame
from tqdm import tqdm

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	pinn = PINN([3, 20, 20, 20, 20, 20, 1], 2.0, device)

	traning_data = generate_traning_data(device, n_colloc=800, n_bc=400, n_ic=200)

	pinn.train(traning_data, 40_000)

	n = solver_options['domain_length']
	iter_end = solver_options['max_iter']
	u = np.zeros((n, n, iter_end))

	for i in tqdm(range(iter_end)):
		temp = np.linspace(0, 1, n)
		X0, Y0 = np.meshgrid(temp, temp)

		X = X0.reshape([n*n, 1])
		Y = Y0.reshape([n*n, 1])

		grid_points = np.concatenate(
			(X, Y,
			np.full((n*n, 1), i * solver_options['delta_t'])),
			axis=1)

		grid_points = torch.tensor(grid_points, dtype=torch.float32).to(device)

		u_pred = pinn.forward(grid_points)
		u_pred = u_pred.cpu().detach().numpy().reshape((n, n))

		u[:, :, i] = u_pred

	animate_plot(u, solver_options['delta_t'])
	plot_frame(u, iter_end-1, solver_options['delta_t']).show()