# src/pinn1d/InfiniteWell/model.py

import math
import torch
import torch.nn as nn


def trig_nodal_factor(x, n):
    s1 = torch.sin(math.pi * x)
    sn = torch.sin(n * math.pi * x)
    ratio = sn / (s1 + 1e-12)
    return torch.where(torch.abs(s1) < 1e-6, torch.full_like(x, float(n)), ratio)


class PINNNet(nn.Module):
    def __init__(self, n=1, hidden=64, use_sine=True):
        super().__init__()
        self.n = n
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
        F   = trig_nodal_factor(x, self.n)
        psi = x * (1.0 - x) * F * out
        return psi


def make_net(n=1, hidden=64, use_sine=True):
    return PINNNet(n=n, hidden=hidden, use_sine=use_sine)
