import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from typing import List, Dict
import time
import numpy as np

class PINN(nn.Module):
	def __init__(self, layers_size: List[int], device, alpha: float = None, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.activation = nn.Tanh()
		self.loss_mse = nn.MSELoss(reduction='mean')

		if alpha is None:
			self.alpha = torch.tensor([1.0], requires_grad=True).to(device)
			self.alpha = nn.Parameter(self.alpha)
		else:
			self.alpha = alpha

		self.device = device

		self.layers = nn.ModuleList([
			nn.Linear(layers_size[i], layers_size[i+1]) for i in range(len(layers_size)-1)
		]).to(device)

		self.init_layers()

	def init_layers(self):
		for i in range(len(self.layers)-1):

			if(self.activation.__class__.__name__ == 'ReLU'):
				nn.init.kaiming_normal_(self.layers[i].weight.data, nonlinearity='relu')
			elif(self.activation.__class__.__name__ == 'Tanh'):
				nn.init.xavier_normal_(self.layers[i].weight.data)

			nn.init.zeros_(self.layers[i].bias.data)

	def forward(self, x: torch.Tensor):
		inp = x
		for i in range(len(self.layers)-1):
			out = self.layers[i](inp)
			inp = self.activation(out)

		out = self.layers[-1](inp)
		return out

	def loss_default(self, x, y):
		return self.loss_mse(self.forward(x), y)

	def loss_pde(self, x_colloc: torch.Tensor):
		x_colloc.requires_grad_()

		x = x_colloc[:, [0]]
		y = x_colloc[:, [1]]
		t = x_colloc[:, [2]]
		u = self.forward(torch.cat([x, y, t], dim=1))

		u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(t).to(self.device), create_graph=True, retain_graph=True)[0]
		u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(x).to(self.device), create_graph=True, retain_graph=True)[0]
		u_y = autograd.grad(u, y, grad_outputs=torch.ones_like(y).to(self.device), create_graph=True, retain_graph=True)[0]
		u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(x).to(self.device), create_graph=True, retain_graph=True)[0]
		u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(y).to(self.device), create_graph=True, retain_graph=True)[0]

		f = u_t - self.alpha * (u_xx + u_yy)

		loss = f.square().mean()

		return loss

	def total_loss(self, x_data, y_data, x_ic, y_ic, x_bc, y_bc, x_colloc) -> torch.Tensor:
		l_data = 0.0
		l_ic = 0.0
		l_bc = 0.0
		l_pde = 0.0

		if x_data is not None and y_data is not None:
			l_data = self.loss_default(x_data, y_data)

		if x_ic is not None and y_ic is not None:
			l_ic = self.loss_default(x_ic, y_ic)

		if x_bc is not None and y_bc is not None:
			l_bc = self.loss_default(x_bc, y_bc)

		if x_colloc is not None:
			l_pde = self.loss_pde(x_colloc)

		return l_data + l_pde + l_ic + l_bc

	def train(self, traning_data: Dict[str, torch.tensor], max_iter: int):
		optimizer = optim.Adam(PINN.parameters(self), lr=2e-4)

		start_time = time.time()
		for i in range(max_iter):
			loss = self.total_loss(**traning_data)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 100 == 0:
				print(f"Iteration: {i}, Loss: {float(loss)}")

		minutes = (time.time() - start_time) / 60
		runtime = round(minutes, 5)
		print(f"Training time: {runtime} min")