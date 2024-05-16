from heat_solver import HeatForwardSolver
import plotter
from solver_params import u0, options, boundaries
import matplotlib.pyplot as plt

if __name__ == "__main__":

	solver = HeatForwardSolver(**options)
	solver.set_initial(u0)
	solver.set_boundaries(boundaries)

	solver.solve()

	#plotter.animate_plot(solver)

	u_label, u_data = solver.generate_traning_data(1000)

	print(f"Labels shape: {u_label.shape}")
	print(f"Data shape: {u_data.shape}")