from pinn import PINN
import torch
import numpy as np
from params import options
from data_generation import generate_traning_data
from plotter import animate_plot, plot_frame
from tqdm import tqdm

if __name__ == "__main__":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	pinn = PINN([3, 40, 40, 40, 40, 40, 40, 40, 1], device, 2.0)

	traning_data = generate_traning_data(device, n_colloc=5000, n_bc=1000, n_ic=400)

	pinn.train(traning_data, 50_000)

	n = options['domain_length']
	iter_end = options['max_iter']
	u = np.zeros((n, n, iter_end))

	for i in tqdm(range(iter_end), desc='Predicting'):
		temp = np.linspace(0, 1, n)
		X0, Y0 = np.meshgrid(temp, temp)

		X = X0.reshape([n*n, 1])
		Y = Y0.reshape([n*n, 1])

		grid_points = np.concatenate(
			(X, Y,
			np.full((n*n, 1), (i * options['delta_t']) / options['t_end'])),
			axis=1)

		grid_points = torch.tensor(grid_points, dtype=torch.float32).to(device)

		u_pred = pinn.forward(grid_points)
		u_pred = u_pred.cpu().detach().numpy().reshape((n, n))

		u[:, :, i] = u_pred

	animate_plot(u, options['delta_t'])
	plot_frame(u, iter_end-1, options['delta_t']).show()