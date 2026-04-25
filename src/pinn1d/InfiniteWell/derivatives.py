import tensorflow as tf

def second_derivative(model, x):
    x = tf.convert_to_tensor(x)
    x = tf.reshape(x, (-1, 1))
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x)
        with tf.GradientTape() as t1:
            t1.watch(x)
            psi = model(x)
        psi_x = t1.gradient(psi, x)
    psi_xx = t2.gradient(psi_x, x)
    return psi, psi_xx
