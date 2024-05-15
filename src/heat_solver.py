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
			  max_iter: float) -> None:

		self.aplha = alpha
		self.delta_x = delta_x
		self.mat_len = round(domain_length / delta_x)
		self.delta_t = delta_t
		self.max_iter = max_iter
		self.boundaries = None

		self.u = np.zeros((self.mat_len, self.mat_len, max_iter), dtype=np.float32)

	def set_initial(self, u0: np.ndarray):
		self.u[:,:,0] = u0

	def set_boundaries(self, boundaries: Callable):
		self.boundaries = boundaries

	def solve(self):
		start_time = time.time()
		const = self.aplha * self.delta_t / self.delta_x**2
		u = self.u

		for k in tqdm(range(1, self.max_iter-1)):

			self.boundaries(u, k, self.delta_t)

			for i in range(1, self.mat_len-1):
				for j in range(1, self.mat_len-1):
					u[i,j,k+1] = const * (u[i+1,j,k] + u[i-1,j,k] + u[i,j+1,k] + u[i,j-1,k] - 4*u[i,j,k]) + u[i,j,k]

		runtime = round(time.time() - start_time, 5)
		print(f"Time: {runtime} seconds")