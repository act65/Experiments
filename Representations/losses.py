import tensorflow as tf
from utils import *

def get_loss_fn(name):
    if name == 'siamese':
        return siamese
    if name == 'orth':
        return orth
    if name == 'ae':
        return ae

def spherical_noise(shape):
    # how to get uniformly distributed noise on a ball?
    z = tf.random_uniform(shape=shape, dtype=tf.float32)
    return z/tf.reduce_mean(z, axis=-1)

def hungarian_matching(x, y):
    # https://github.com/tensorflow/tensorflow/pull/3780
    # could compile binary and add op
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
        return scale*tf.reduce_mean(tf.square(inputs - z))

def mutual_info(inputs):
    """
    Noise as targets can be interpreted as optimising the mutual information
    of ???
    """
    return None


def siamese(inputs, scale):
    """
    Want each datapoint to be a distance of 1 away form every other datapoint.

    Instead of running two networks side by side we can just split the batch
    into two parts achieving the same result.
    Or just use the whole rest of the batch?!

    min sum_i sum_j 1-d(x_i, y_i))

    """
    # NOTE. How is this and orthogonal regularisation different?
    # But are minimising 1 - the distance between datapoints.
    with tf.name_scope('equidist_regulariser'):
        inputs = tf.reduce_mean(inputs, axis=[1,2])
        batch_size = tf.shape(inputs)[0]
        diff = tf.expand_dims(inputs, 0) - tf.expand_dims(inputs, 1)
        similarities = tf.sqrt(1e-8+tf.reduce_mean(tf.square(diff), axis=-1))
        targets = 1-tf.eye(batch_size)
        loss_val = scale*tf.reduce_mean(tf.square(targets-similarities))

        return loss_val


def gan(inputs, scale):
    """
    We have no idea what type of regularisation is a good idea for unsupervised
    pretraining (for discrimination).
    Instead can we learn the right function?
    Our true goal is: ??
    - easily discriminable datapoints (aka uniformly distributed?)
    - disentangled representations (aka clustered?)
    - ?
    """
    with tf.name_scope(name):
        z = spherical_noise(inputs.get_shape())
        fake = discriminator(inputs)
        real = discriminator(z)

        loss_val = adversarial(fake, real)

def ae(inputs, scale, name='autoencoder'):
    """
    How are representations learn by AEs different from other methods?
    How can we measure this difference?
    - ML, explainable variance, orthogonality, distance between data,
    (if we knew this, then we could just optimise it...)
    """
    with tf.name_scope(name):
        x = tf.get_collection('inputs')
        y = decoder(inputs)
        loss_val = tf.reduce_mean(tf.square(x-y))

        return loss_val


def kl(inputs, scale):
    """
    A weird idea. https://arxiv.org/abs/1705.00574
    Lacking a theoretical motivation (doesnt mean it doesnt have one).
    Interpret output representation as logits, and use to create a probability
    distribution over outputs (as binary variables?).
    Maximise the difference (measured by KL) in distributions between each
    datapoint.
    """
    # NOTE. How is this and orthogonal regularisation different?
    with tf.name_scope('kl_regulariser'):
        batch_size = tf.shape(inputs)[0]
        probs = tf.nn.softmax(inputs)
        diff = tf.expand_dims(probs, 0) + tf.expand_dims(probs, 1)
        similarities = tf.sqrt(1e-8+tf.reduce_mean(tf.square(diff), axis=-1))
        cond = 1-tf.eye(batch_size)
        loss_val = scale*tf.reduce_mean(tf.square(cond*similarities))

        return loss_val


def orth(inputs, scale, normalise=False, summarise=True,
                           name='orthogonal_regulariser'):
    """
    Regulariser to enourage a batch of things to be orthonormal.
    Aka, let x be [batch, -1] flattened version of inputs,
    we return ||xx^T - I||^2

    Note that due to the reshaping required this will only work if the batch
    size is defined.

    Closely related to:
        - cross correlation regularisation
            - [DeCov](https://arxiv.org/pdf/1511.06068.pdf) and
            - [Xcov](https://arxiv.org/pdf/1412.6583.pdf)
        - negative sampling. [word2vec](https://arxiv.org/pdf/1402.3722v1.pdf)
            maximises/minimises similarity (measured by cosine distance).
            The negative sample is the rest of the batch.

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
        inputs = tf.reduce_mean(inputs, axis=[1,2])
        batch_size = inputs.get_shape().as_list()[0]
        inputs = tf.reshape(inputs, [batch_size, -1])
        if normalise:
            inputs = tf.nn.l2_normalize(inputs, -1)
        similarities = tf.matmul(inputs, inputs, transpose_b=True)
        targets = tf.eye(batch_size)
        loss_val = scale*tf.reduce_mean(tf.square(similarities - targets))

        if summarise:
            tf.summary.image('similarities',
                             tf.reshape(similarities,
                                        [1, batch_size, batch_size, 1]))
            tf.summary.scalar('orthogonal_reg', loss_val)
    return loss_val
