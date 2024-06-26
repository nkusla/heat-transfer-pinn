import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
from typing import List
from data_generation import inverse_scale_alpha

def plot_frame(u: np.ndarray, k: int, delta_t: float, pbar: tqdm = None):
	plt.clf()
	plt.title(f"t = {round(k * delta_t, 4)}")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.pcolormesh(u[:,:,k], cmap=plt.cm.jet)
	plt.colorbar(label="u(x, y, t)")

	if pbar is not None:
		pbar.update(1)

	return plt

def animate_plot(u: np.ndarray, delta_t: float, fps: int = 30, filename: str = "HeatAnimation.gif"):
	fig = plt.figure()

	max_iter = u.shape[2]

	pbar = tqdm(total=max_iter, desc="Generating animation")

	anim = FuncAnimation(
		fig,
		lambda k: plot_frame(u, k, delta_t, pbar),
		frames=max_iter,
		interval=1,
		repeat=True)

	anim.save(filename, fps=fps)
	print(f"Plot animated and saved to {filename}")

	return anim

def plot_loss(loss_history: List[float], epoch_history: List[float]):
	plt.plot(epoch_history, loss_history)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss vs Epoch")
	plt.grid()

	return plt

def plot_alpha(alpha_history: List[float], epoch_history: List[float], true_alpha: float):
	alpha_history = list(map(inverse_scale_alpha, alpha_history))
	plt.plot(epoch_history, alpha_history)
	plt.xlabel("Epoch")
	plt.ylabel("Alpha")
	plt.title("Alpha vs Epoch")
	plt.grid()
	plt.axhline(y=true_alpha, color='r', linestyle='--', label="true alpha")
	plt.legend()

	return plt