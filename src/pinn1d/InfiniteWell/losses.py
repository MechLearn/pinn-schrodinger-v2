# src/pinn1d/InfiniteWell/losses.py

import torch
from .derivatives import second_derivative


def compute_losses(model, x_batch, alpha, lam):
    """
    Pérdida para el pozo infinito 1D.
    Residuo: ψ'' + E·ψ = 0  con V=0
    """
    psi, psi_xx = second_derivative(model, x_batch)

    E = torch.nn.functional.softplus(alpha) + 1e-8

    res  = psi_xx + E * psi
    LPDE = torch.mean(res ** 2)

    # Normalización por trapecio
    psi2     = psi.squeeze() ** 2
    xb       = x_batch.squeeze()
    dx       = xb[1:] - xb[:-1]
    integral = torch.sum(0.5 * (psi2[1:] + psi2[:-1]) * dx)
    Lnorm    = (integral - 1.0) ** 2

    L = LPDE + lam * Lnorm
    return L, LPDE, Lnorm, integral, E