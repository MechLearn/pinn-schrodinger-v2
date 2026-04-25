# src/pinn1d/Harmonic/losses.py

import tensorflow as tf
from .derivatives import second_derivative


@tf.function
def compute_losses(net, x_batch, alpha, lam):
    """
    Pérdida para el oscilador armónico 1D.

    Ecuación: -ψ'' + (x²/2) ψ = E ψ
    Residuo:   ψ'' + (E - x²/2) ψ = 0
    """
    psi, psi_xx = second_derivative(net, x_batch)

    E = tf.nn.softplus(alpha) + 1e-8

    x = tf.reshape(x_batch, (-1, 1))
    V = 0.5 * tf.square(x)

    res  = psi_xx + (E - V) * psi
    LPDE = tf.reduce_mean(tf.square(res))

    psi2     = tf.squeeze(tf.square(psi), axis=1)
    xb       = tf.squeeze(tf.convert_to_tensor(x_batch), axis=1)
    dx       = xb[1:] - xb[:-1]
    integral = tf.reduce_sum(0.5 * (psi2[1:] + psi2[:-1]) * dx)
    Lnorm    = tf.square(integral - 1.0)

    L = LPDE + lam * Lnorm
    return L, LPDE, Lnorm, integral, E