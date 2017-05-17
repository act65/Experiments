import tensorflow as tf
from utils import *

import munkres

def get_loss_fn(name, h):
    if name == 'siamese':
        return siamese(tf.reduce_mean(h, [1,2]), 1.0)
    if name == 'spaced':
        return spaced(tf.reduce_mean(h, [1,2]), 1.0)
    if name == 'orth':
        return orth(tf.reduce_mean(h, [1,2]), 1.0)
    if name == 'ae':
        return ae(h)
    if name == 'nat':
        return nat(tf.reduce_mean(h, [1,2]), 1.0)

def nat(inputs, scale, normalise=True, name='noiseastargets'):
    """
    [Noise as targets](https://arxiv.org/abs/1704.05310).
    This attempts to train our hidden representation, y, to be similar to a
    normal distribution over the unit ball. This is achineved by matching
    random noise to the closest hidden representations of a batch.
    min d(x, noise) Thus the name; Noise as targets.

    (What about training to be indistinguishable from a uniform distribution? GANs)
    Or using a different measure of distance? MMD
    """
    # TODO. try without using the matching algorithm!!!
    # TODO. try with different distributions of noise?!

    # NOTE. what is the effect of batch size? as with a larger batch it would be
    # easier to find a closer target.
    def spherical_noise(shape):
        with tf.name_scope('spherical_noise'):
            z = tf.random_uniform(shape=shape, dtype=tf.float32)
            return tf.nn.l2_normalize(z, -1)

    def spherical_variable_noise(shape):
        # a version of noise that gets updated only when called.
        with tf.name_scope('spherical_var_noise'):
            z = tf.get_variable(name='noise_var', trainable=False,
                                initializer=tf.random_normal_initializer(),
                                shape=[shape[1], shape[1]], dtype=tf.float32)
            tf.add_to_collection('random_vars', z)
            return tf.nn.l2_normalize(z, -1)

    def hungarian_matching(h, z):
        """Hungarian matching algorithm. O(n^3)

        Alternative approaches.
        - could compile binary and add op
        https://github.com/tensorflow/tensorflow/pull/3780
        - could just wrap a python op. will be slower. (but easier)
        """
        M = munkres.Munkres()

        def get_pairings(C):
            """
            An implementation using munkres python library and py_wrap.
            Args:
                C (np.ndArray): the costs of different pairings

            Returns:
                list: new pairings
            """
            # TODO. this is WAY TOO slow. but maybe its just the hungarian algol?
            assignments = M.compute(C)
            return np.array(zip(*assignments))

        def cost_fn(x, y):
            with tf.name_scope('matching_cost'):
                return tf.reduce_mean((tf.expand_dims(x, 1) -
                                       tf.expand_dims(y, 0))**2, axis=2)

        with tf.name_scope('matching'):
            C = cost_fn(h , z)
            idx = tf.py_func(get_pairings, [C], tf.int64)
            idx = tf.reshape(idx, [2, tf.shape(h)[0]])
            return tf.gather(h, idx[0]), tf.gather(z, idx[1])

    with tf.name_scope(name):
        z = spherical_variable_noise(inputs.get_shape())
        features, targets = hungarian_matching(inputs, z)
        if normalise:
            features = tf.nn.l2_normalize(features, -1)
        return scale*tf.reduce_mean(tf.square(features - targets))

def pairwise_l2_dist(x, y):
    # could do this better. only need to calculate the upper/lower triangular?
    diff = tf.expand_dims(x, 0) - tf.expand_dims(y, 1)
    return tf.sqrt(1e-8+tf.reduce_mean(tf.square(diff), axis=-1))

def spaced(inputs, scale, name='spaced'):
    """
    Want each datapoint to be a distance of 1 away form every other datapoint.
    We are regularising the density of points on the output space to be
    of uniform density.


    Instead of running two networks side by side we can just split the batch
    into two parts achieving the same result.
    Or just use the whole rest of the batch?!

    min sum_i sum_j 1-d(x_i, y_i))  -- i!=j
    """
    # NOTE. Uniform distance to batch_num of points gives us some sort of weird
    # high dimensional polygon? Is it even possible to be uniformly spaced == 1
    # for 50 points?
    # NOTE. would prefer. a grid. am dist = 1 from my nearest neighbors.

    # NOTE. How is this and orthogonal regularisation different?
    # Both are minimising 1 - distance between datapoints.
    with tf.name_scope(name):
        batch_size = tf.shape(inputs)[0]
        distances = pairwise_l2_dist(inputs, inputs)
        targets = 1 - tf.eye(batch_size)
        loss_val = scale*tf.reduce_mean(tf.square(targets-distances))

        return loss_val

def grid(inputs, scale, name='grid'):
    """
    Want each datapoint to be a distance of 1 away form its nearest neighbors.
    We are regularising the density of points on the output space to be
    of uniform density.
    """
    def bottom_k(x, k=1):
        with tf.name_scope('bottom_k'):
            d = tf.shape(x)[-1]
            x = tf.reshape(x, [-1,d])
            x += 1000*tf.eye(d) # hack to make sure not self-similar
            x = -(x**2)  # use top k to get bottom k...
            values, indices = tf.nn.top_k(x, k=k, sorted=False)
            # indexing x is too painful. just cheat
            return tf.sqrt(tf.abs(values) + 1e-8)

    with tf.name_scope(name):
        batch_size = tf.shape(inputs)[0]
        distances = pairwise_l2_dist(inputs, inputs)
        k_distances = bottom_k(distances, 4)  # local density
        targets = tf.ones_like(k_distances)  # uniformly spaced
        loss_val = scale*tf.reduce_mean(tf.square(targets-k_distances))
        return loss_val


# TODO. something like what t-sne uses would be good?!
def siamese(inputs, scale, name='siamese'):
    """
    Inputs that are similar should be mapped to outputs that are similar.
    L2 doesnt seem like the right measure for this...

    This has some pretty degenerate cases.
    Translate an input
    """
    with tf.name_scope(name):
        x = tf.reduce_mean(tf.get_collection('inputs')[0], axis=[1,2])
        # TODO. averaging over the spatial indexes when the number of features
        # is only 1 doesnt make much sense... we are just matching the means?!
        distances = pairwise_l2_dist(inputs, inputs)
        targets = pairwise_l2_dist(x, x)
        return scale*tf.reduce_mean(tf.square(targets-distances))
        # should use some more like the opposite of x**2? kinda logistic?
        # want a high loss untill very close.

def cluster(inputs, scale, name='cluster'):
    """
    A loss that encourages close points to be closer.
    """

    with tf.name_scope(name):
        return 1

def gan(inputs, scale, name='gan'):
    """
    We have no idea what type of regularisation is a good idea for unsupervised
    pretraining (for discrimination).
    Instead can we learn the right function?
    Our true goal is: ??
    - easily discriminable datapoints (aka uniformly distributed?)
    - disentangled representations (aka ??)
    - ?
    """
    def discriminator(x):
        return None

    def adversarial(fake, real):
        return (tf.reduce_mean(tf.log(fake)),
                tf.reduce_mean(-tf.log(fake) + tf.log(real)))

    with tf.name_scope(name):
        z = spherical_noise(inputs.get_shape())
        fake = discriminator(inputs)
        real = discriminator(z)

        loss_val = adversarial(fake, real)
        # TODO. but we also need to train the discriminator. how?

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

    This is (almost) the same as regularising the weights, but way cheaper.
    If W is orthogonal (and x is orthogonal) then, if y=Wx, y is orthogonal.

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
        targets = tf.eye(batch_size)
        loss_val = scale*tf.reduce_mean(tf.square(similarities - targets))

        if summarise:
            tf.summary.image('similarities',
                             tf.reshape(similarities,
                                        [1, batch_size, batch_size, 1]))
            tf.summary.scalar('orthogonal_reg', loss_val)
    return loss_val
