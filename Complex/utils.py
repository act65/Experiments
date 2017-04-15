import tensorflow as tf
import tensorflow.contrib.slim as slim

def get_complex_var(name, shape, dtype, initializer):
    a = tf.get_variable(name=name+'a', shape=shape, dtype=tf.float32, initializer=initializer)
    b = tf.get_variable(name=name+'b', shape=shape, dtype=tf.float32, initializer=initializer)
    return tf.complex(a, b)

@slim.add_arg_scope
def complex_fc(x, num_outputs, weights_initializer, biases_initializer=None, activation_fn=None, scope=''):
    n = x.get_shape().as_list()[-1]
    with tf.variable_scope('fc'+scope):
        W = get_complex_var(name='W', shape=[n, num_outputs],
                            dtype=tf.complex64,
                            initializer=weights_initializer)
        y = tf.matmul(x, W)
        if biases_initializer:
            b = get_complex_var(name='b', shape=[1, num_outputs],
                                dtype=tf.complex64,
                                initializer=biases_initializer)
            y += b
        if activation_fn:
            return activation_fn(y)
        else:
            return y
