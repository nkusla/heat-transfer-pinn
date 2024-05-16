import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heat_solver import HeatForwardSolver
import numpy as np

def plot_frame(u: np.ndarray, k: int, delta_t: float):
	plt.clf()
	plt.title(f"t = {round(k * delta_t, 4)}")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.pcolormesh(u[:,:,k], cmap=plt.cm.jet)
	plt.colorbar(label="u(x, y, t)")
	return plt

def animate_plot(u: np.ndarray, delta_t: float):
	fig = plt.figure()

	max_iter = u.shape[2]

	anim = FuncAnimation(
		fig,
		lambda k: plot_frame(u, k, delta_t),
		frames=max_iter,
		interval=1,
		repeat=True)

	anim_path = "HeatAnimation.gif"
	anim.save(anim_path, fps=60)
	print(f"Plot animated and saved to {anim_path}")