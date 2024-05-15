from heat_solver import HeatForwardSolver
import plotter
from params import u0, options, boundaries
import matplotlib.pyplot as plt


if __name__ == "__main__":

	solver = HeatForwardSolver(**options)
	solver.set_initial(u0)
	solver.set_boundaries(boundaries)

	solver.solve()

	plotter.animate_plot(solver)