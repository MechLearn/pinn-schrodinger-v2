import math
import tensorflow as tf

def trig_nodal_factor(x, n):
    s1 = tf.sin(math.pi * x)
    sn = tf.sin(n * math.pi * x)
    ratio = sn / (s1 + 1e-12)
    return tf.where(tf.abs(s1) < 1e-6, tf.cast(n, x.dtype), ratio)

def make_net(n=1, hidden=64, use_sine=True):
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

    F = tf.keras.layers.Lambda(lambda x: trig_nodal_factor(x, n))(x_in)
    psi = x_in * (1.0 - x_in) * F * out
    return tf.keras.Model(inputs=x_in, outputs=psi)
