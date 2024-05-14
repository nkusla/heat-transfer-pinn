import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heat_solver import HeatForwardSolver
import numpy as np

def plot_frame(solver: HeatForwardSolver, k: int, vmin: float, vmax: float):
	plt.clf()
	plt.title(f"t = {round(k * solver.delta_t, 4)}")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.pcolormesh(solver.u[:,:,k], cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
	plt.colorbar(label="u(x, y, t)")
	return plt

def animate_plot(solver: HeatForwardSolver):
	fig = plt.figure()

	vmax = np.max(solver.u.flatten())
	vmin = np.min(solver.u.flatten())

	anim = FuncAnimation(
		fig,
		lambda k: plot_frame(solver, k, vmin, vmax),
		frames=solver.max_iter,
		interval=0.2,
		repeat=True)

	anim.save("anim.gif")