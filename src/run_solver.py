from heat_solver import HeatForwardSolver
import plotter
from solver_params import u0, solver_options, boundaries
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
	np.random.seed(1616)

	solver = HeatForwardSolver(**solver_options)
	solver.set_initial(u0)
	solver.set_boundaries(boundaries)

	solver.solve()

	plotter.animate_plot(solver.u, solver.delta_t)
	plotter.plot_frame(solver.u, solver.u.shape[2]-1, solver.delta_t).show()

	u_label, u_data = solver.generate_traning_data(1000)

	print(f"Labels shape: {u_label.shape}")
	print(f"Data shape: {u_data.shape}")