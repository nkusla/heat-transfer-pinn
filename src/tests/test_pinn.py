import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pinn import PINN
import torch
import numpy as np
from params import options
from data_generation import generate_traning_data, scale_alpha
from plotter import animate_plot, plot_frame
from tqdm import tqdm
from heat_solver import HeatForwardSolver
import plotter
from params import u0, options, boundaries
from predict import predict

if __name__ == "__main__":
	solver = HeatForwardSolver(**options)
	solver.set_initial(u0)
	solver.set_boundaries(boundaries)

	solver.solve()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	scaled_alpha = scale_alpha(options['alpha'])
	pinn = PINN([3, 40, 40, 40, 40, 40, 40, 40, 1], device, scaled_alpha)

	traning_data = generate_traning_data(device,
		u=solver.u, n_data=20_000)

	pinn.start_train(traning_data, 20_000)

	u = predict(pinn)
