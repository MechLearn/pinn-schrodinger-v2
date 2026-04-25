import tensorflow as tf
from .derivatives import second_derivative

def inv_softplus(y):
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return tf.math.log(tf.math.expm1(y) + 1e-12)

@tf.function
def compute_losses(net, x_batch, alpha, lam):
    psi, psi_xx = second_derivative(net, x_batch)

    E = tf.nn.softplus(alpha) + 1e-8         # energía positiva
    res = psi_xx + E * psi                   # V=0 → ψ'' + Eψ = 0
    LPDE = tf.reduce_mean(tf.square(res))

    # Normalización por trapecio (igual a tu notebook)
    psi2 = tf.squeeze(tf.square(psi), axis=1)
    xb = tf.squeeze(tf.convert_to_tensor(x_batch), axis=1)
    dx = xb[1:] - xb[:-1]
    integral = tf.reduce_sum(0.5*(psi2[1:]+psi2[:-1])*dx)
    Lnorm = tf.square(integral - 1.0)

    L = LPDE + lam * Lnorm
    return L, LPDE, Lnorm, integral, E
