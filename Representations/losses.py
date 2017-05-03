import tensorflow as tf

def spherical_noise(shape):
    # how to get uniformly distributed noise on a ball?
    z = tf.random_uniform(shape=shape, dtype=tf.float32)
    return z/tf.reduce_mean(z, axis=-1)

def hungarian_matching(x, y):
    return x, y

def nat(inputs, scale):
    """
    [Noise as targets](https://arxiv.org/abs/1704.05310).
    This attempts to train our hidden representation, y, to be similar to a
    normal distribution over the unit ball. This is achineved by matching
    random noise to the closest hidden representations of a batch.
    min d(x, noise) Thus the name; Noise as targets.

    (What about training to be indistinguishable from a uniform distribution? GANs)
    Or using a different measure of distance? MMD

    Args:
        inputs:


    """
    with tf.name_scope(name):
        z = spherical_noise(inputs.get_shape())
        inputs, z = hungarian_matching(inputs, z)
        return scale*tf.nn.l2_loss(inputs - z)

def siamese(y):
    """

    """
    n = tf.shape(y)[0]
    a = y[:n]
    b = y[n:]
    return tf.reduce_mean(1-tf.norm(a-b))

def orth(inputs, scale, normalise=False, summarise=True,
                           name='orthogonal_regulariser'):
    """Regulariser to enourage a batch of things to be orthonormal.
    Aka, let x be [batch, -1] flattened version of inputs,
    we return ||xx^T - I||^2

    Note that due to the reshaping required this will only work if the batch
    size is defined.

    Args:
        inputs: tensor of inputs, can be any shape as long as the batch is the
            first dimension.
        scale: scalar multiplier to control the amount of regularisation.
        normalise (Optional[bool]): whether we should normalise the inputs or
            not. If false, then this regulariser does two things -- pushes
            the vectors towards unit (l2) norm as well as forcing
            dissimilarity.
        summarise (Optional[bool]): if true, we will add an image summary
            showing the dot products and a scalar summary with the value.
        name (Optional[str]): name for the name scope under which the ops
            are constructed.

    Returns:
        scalar tensor holding the value to minimise.
    """
    with tf.name_scope(name):
        batch_size = inputs.get_shape().as_list()[0]
        inputs = tf.reshape(inputs, [batch_size, -1])
        if normalise:
            inputs = tf.nn.l2_normalize(inputs, -1)
        similarities = tf.matmul(inputs, inputs, transpose_b=True)
        diffs = similarities - tf.eye(batch_size)
        loss_val = scale * tf.reduce_mean(tf.square(diffs))

        if summarise:
            tf.summary.image('similarities',
                             tf.reshape(similarities,
                                        [1, batch_size, batch_size, 1]))
            tf.summary.image('difference_from_eye',
                             tf.reshape(diffs,
                                        [1, batch_size, batch_size, 1]))
            tf.summary.scalar('orthogonal_reg', loss_val)
    return loss_val


# TODO. actualy cross entropy. what actual discriminative loss... 
