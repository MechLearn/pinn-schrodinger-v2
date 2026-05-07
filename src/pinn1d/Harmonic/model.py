# src/pinn1d/Harmonic/model.py

import torch
import torch.nn as nn


class PINNNet(nn.Module):
    def __init__(self, n=1, hidden=64, use_sine=True):
        super().__init__()
        self.n   = n
        self.act = torch.sin if use_sine else torch.tanh

        self.layer1 = nn.Linear(1, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.out    = nn.Linear(hidden, 1)

        for layer in [self.layer1, self.layer2, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        z   = self.act(self.layer1(x))
        z   = self.act(self.layer2(z))
        out = self.out(z)

        # Decaimiento gaussiano — condición de frontera
        gauss = torch.exp(-0.5 * x ** 2)

        # Paridad forzada
        if self.n % 2 == 1:
            psi = gauss * out
        else:
            psi = gauss * x * out

        return psi


def make_net(n=1, hidden=64, use_sine=True, x_max=6.0):
    return PINNNet(n=n, hidden=hidden, use_sine=use_sine)
