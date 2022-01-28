import torch
import torch.nn as nn

ACTIVATIONS = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SineOm(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)
