# src/pinn1d/Harmonic/model.py

import tensorflow as tf


def make_net(n=1, hidden=64, use_sine=True, x_max=6.0):
    """
    Ansatz para el oscilador armónico 1D.

    Filosofía idéntica al pozo infinito:
    - exp(-x²/2): condición de frontera por construcción (equivalente a x(1-x))
    - Paridad forzada: guía espectral (equivalente a sin(nπx)/sin(πx))
        n impar (1,3,5,...) → eigenfunción PAR  → ψ = exp(-x²/2) · N(x; θ)
        n par   (2,4,6,...) → eigenfunción IMPAR → ψ = exp(-x²/2) · x · N(x; θ)
    - N(x; θ): red neuronal que aprende la corrección fina
    """
    x_in = tf.keras.Input(shape=(1,))

    if use_sine:
        z = tf.keras.layers.Dense(hidden, activation=tf.math.sin,
                                  kernel_initializer="glorot_uniform",
                                  bias_initializer="zeros")(x_in)
        z = tf.keras.layers.Dense(hidden, activation=tf.math.sin,
                                  kernel_initializer="glorot_uniform",
                                  bias_initializer="zeros")(z)
    else:
        z = tf.keras.layers.Dense(hidden, activation="tanh",
                                  kernel_initializer="glorot_uniform",
                                  bias_initializer="zeros")(x_in)
        z = tf.keras.layers.Dense(hidden, activation="tanh",
                                  kernel_initializer="glorot_uniform",
                                  bias_initializer="zeros")(z)

    out = tf.keras.layers.Dense(1, activation=None,
                                kernel_initializer="glorot_uniform",
                                bias_initializer="zeros")(z)

    # Decaimiento gaussiano — equivalente a x(1-x) del pozo infinito
    gauss = tf.keras.layers.Lambda(
        lambda x: tf.exp(-0.5 * tf.square(x))
    )(x_in)

    # Paridad forzada — equivalente al envelope espectral del pozo infinito
    if n % 2 == 1:
        # n impar → eigenfunción par
        psi = gauss * out
    else:
        # n par → eigenfunción impar
        psi = gauss * x_in * out

    return tf.keras.Model(inputs=x_in, outputs=psi)