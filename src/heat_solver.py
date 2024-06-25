import numpy as np
from typing import Tuple, Callable
from tqdm import tqdm
import time

class HeatForwardSolver():
	def __init__(self,
			alpha: float,
			delta_x: float,
			domain_length: float,
			delta_t: float,
			max_iter: float,
			*args, **kwargs) -> None:

		self.aplha = alpha
		self.delta_x = delta_x
		self.mat_len = round(domain_length / delta_x)
		self.delta_t = delta_t
		self.max_iter = max_iter
		self.boundaries = None

		assert delta_t <= (delta_x ** 2 / (2*alpha)), "This solver config will produce unstable solutions"

		self.u = np.zeros((self.mat_len, self.mat_len, max_iter), dtype=np.float32)

	def set_initial(self, u0: np.ndarray):
		self.u[:,:,0] = u0

	def set_boundaries(self, boundaries: Callable):
		self.boundaries = boundaries

	def solve(self):
		start_time = time.time()
		const = self.aplha * self.delta_t / self.delta_x**2
		u = self.u

		for k in tqdm(range(1, self.max_iter-1), desc="Running solver"):

			self.boundaries(u, k, self.delta_t)

			for i in range(1, self.mat_len-1):
				for j in range(1, self.mat_len-1):
					u[i,j,k+1] = const * (u[i+1,j,k] + u[i-1,j,k] + u[i,j+1,k] + u[i,j-1,k] - 4*u[i,j,k]) + u[i,j,k]

		runtime = round(time.time() - start_time, 5)
		print(f"Time: {runtime} seconds")

	def generate_traning_data(self, num_points):
		print("Generating traning data")

		u = self.u
		u_label = np.empty(num_points, dtype=np.float32)
		u_data = np.empty((num_points, 3), dtype=np.float32)

		rand_pos = np.random.randint(0, self.mat_len-1, size=(num_points, 2))
		rand_t = np.random.randint(0, self.max_iter-1, size=(num_points, 1))

		rand_idx = np.hstack((rand_pos, rand_t))

		for i, rand in enumerate(tqdm(rand_idx)):
			u_label[i] = u[tuple(rand)]

			data = np.array([
				rand[0] * self.delta_x,
				rand[1] * self.delta_x,
				rand[2] * self.delta_t
			], dtype=np.float32)

			u_data[i] = data

		return (u_label, u_data)