import tensorflow as tf

def nat(y):
    """

    """
    z = tf.random_normal()
    y, z = match(y, z)
    return tf.l2_loss(y - z)

def siamese(y):
    """

    """
    n = tf.shape(y)[0]
    a = y[:n]
    b = y[n:]
    return tf.reduce_mean(1-tf.norm(a-b))
