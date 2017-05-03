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

def orthogonal_regulariser(inputs, scale, normalise=False, summarise=True,
                           name='orthogonality_constraint'):
    """Regulariser to enourage a batch of things to be orthonormal.
    Aka, let x be [batch, -1] flattened version of inputs,
    we return ||xx^T - I||^2

    Note that due to the reshaping required this will only worj if the batch
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
