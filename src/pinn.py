import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from typing import List, Dict
import time
from tqdm import tqdm
import numpy as np

torch.manual_seed(1616)

class PINN(nn.Module):
	def __init__(self, layers_size: List[int], alpha, device, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.activation = nn.Tanh()
		self.loss_mse = nn.MSELoss(reduction='mean')

		self.alpha = alpha
		self.device = device

		self.layers = nn.ModuleList([
			nn.Linear(layers_size[i], layers_size[i+1]) for i in range(len(layers_size)-1)
		]).to(device)

		for i in range(len(self.layers)-1):
			nn.init.xavier_normal_(self.layers[i].weight.data)
			nn.init.zeros_(self.layers[i].bias.data)

	def forward(self, x: torch.Tensor):
		inp = x
		for i in range(len(self.layers)-2):
			out = self.layers[i](inp)
			inp = self.activation(out)

		out = self.layers[-1](inp)
		return out

	def loss_data(self, x_data, y_data):
		return self.loss_mse(self.forward(x_data), y_data)

	def loss_bc(self, x_bc, y_bc):
		return self.loss_mse(self.forward(x_bc), y_bc)

	def loss_pde(self, x_colloc: torch.Tensor):
		x_colloc.requires_grad_()

		x = x_colloc[:, [0]]
		y = x_colloc[:, [1]]
		t = x_colloc[:, [2]]
		u = self.forward(torch.cat([x, y, t], dim=1))

		u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True, retain_graph=True)[0]
		u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True, retain_graph=True)[0]
		u_y = autograd.grad(u, y, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True, retain_graph=True)[0]
		u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x).to(self.device), create_graph=True, retain_graph=True)[0]
		u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y).to(self.device), create_graph=True, retain_graph=True)[0]

		f = u_t - self.alpha * (u_xx + u_yy)

		loss = self.loss_mse(f, torch.zeros_like(f))

		return loss

	def total_loss(self, x_data, y_data, x_bc, y_bc, x_colloc) -> torch.Tensor:
		l_data = 0.0
		l_bc = 0.0
		l_pde = 0.0

		if x_data is not None and y_data is not None:
			l_data = self.loss_data(x_data, y_data)

		if x_bc is not None and y_bc is not None:
			l_bc = self.loss_bc(x_bc, y_bc)

		if x_colloc is not None:
			l_pde = self.loss_pde(x_colloc)

		return l_pde + l_bc + l_data

	def train(self, traning_data: Dict[str, np.ndarray], max_iter: int, optimizer: torch.optim.Optimizer = None):
		if optimizer is None:
			optimizer = optim.Adam(PINN.parameters(self), lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

		for i in range(max_iter):
			loss = self.total_loss(**traning_data)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 100 == 0:
				print(f"Iteration: {i}, Loss: {float(loss)}")