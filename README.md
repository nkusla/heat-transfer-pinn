# Simulation of heat propagation using Physics-informed Neural Network

This project applies Physics-informed Neural Networks (PINNs) to simulate heat propagation, comparing it with the traditional finite difference method. It demonstrates PINNs' capability in solving the heat equation PDE given certain boundary and initial conditions. The comparison is visually represented through GIFs in the README, showcasing the effectiveness of PINNs in modeling thermal dynamics. This project also explores the use of PINN for PDE discovery by learning the thermal diffusivity (alpha) of materials, showcasing its potential in identifying unknown parameters in physical laws.

### Examples

| Finite difference method | Physics-informed Neural Network |
| :---: | :---: |
| ![](plots/numerical_prediction.gif) | ![](plots/pinn_prediction_all.gif) |

### Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Code examples and neural network training is located in `main.ipynb` notebook.