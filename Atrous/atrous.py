from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np
import cv2

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.app.flags.DEFINE_bool('atrous', False, 'whether to use atrous conv')
tf.app.flags.DEFINE_bool('scale', True, 'whether to randomly scale data')
tf.app.flags.DEFINE_bool('same_params', True, 'whether to use same n of params '
                         'vs same amount of compute.')
FLAGS = tf.app.flags.FLAGS


def batch(ims, labels, batchsize):
    ims, labels = shuffle(ims, labels)
    shape = ims.shape
    for i in range(len(labels)//batchsize):
        yield (i, ims[i*batchsize:(i+1)*batchsize, ...],
               labels[i*batchsize:(i+1)*batchsize, ...])


def random_scale(x):
    y = []
    shape = x.shape
    for im in x:
        i = np.random.randint(0, 50)
        padded = np.pad(im, [[i,i],[i,i],[0,0]], 'constant')
        y.append(cv2.resize(padded, (28, 28)))
    return np.stack(y, axis=0).reshape(shape)


# def randomscale(x):
#     # random sized padding and resize
#     z = []
#     shape = x.get_shape().as_list()
#     for im in tf.unstack(x, axis=0):
#         i = tf.random_uniform(shape=[], dtype=tf.int32, maxval=20, minval=0)
#         y = tf.pad(im, [[i, i], [i, i], [0,0]], "CONSTANT")
#         # images will be in the center. what about stretching in the x or y dim?
#         z.append(tf.image.resize_images(y, [shape[1], shape[2]]))
#     z = tf.stack(z, axis=0)
#     z.set_shape(shape)
#     return z


@slim.add_arg_scope
def multiscale_atrousconv(x, channels, filter_size=3, n=4, activation_fn=tf.nn.relu,
                          weights_initializer=None, bias_initializer=None, scope=''):
    """
    Args:
        x (tf.Tensor): a tensor of input values. [batch, width, height, k]
        channels (int): number of feautres to output
        filter_size (int): number of params to use into each direction.
        n (int): number of different scales (defaults to powers of 2) to convolve over.

    Returns:
        y (tf.Tensor): outputs. [batch, width, height, channels]
    """
    # it feels weird how we cannot down sample
    shape = x.get_shape().as_list()
    with tf.variable_scope('multi_scale_conv' + scope):
        filters = tf.get_variable(shape=[filter_size, filter_size, shape[-1],
                                  channels//n], dtype=tf.float32,
                                  name='filters', initializer=weights_initializer)
        bias = tf.get_variable(shape=[channels] , dtype=tf.float32, name='bias',
                               initializer=bias_initializer)
        y = tf.concat(3, [tf.nn.atrous_conv2d(x, filters, (i+1), padding='SAME')
                          for i in range(n)])
        return activation_fn(y + bias)


def accuracy(p, y):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p, axis=1), y), tf.float32))

def main(_):
    print('Get data')
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=False)
    ims = np.reshape(mnist.train.images, [-1, 28, 28, 1]).astype(np.float32)
    labels = np.reshape(mnist.train.labels, [-1]).astype(np.int64)

    test_ims = np.reshape(mnist.test.images, [-1, 28, 28, 1]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)

    x = tf.placeholder(shape=[50, 28, 28, 1], dtype=tf.float32)
    y = tf.placeholder(shape=[50], dtype=tf.int64)
    # if FLAGS.scale:
    #     x = randomscale(x)


    # channel_sizes = [16, 16, 16, 16]
    channel_sizes = [32, 32, 32]
    n = 4

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
        # TODO need BN?
        if FLAGS.atrous:
            if not FLAGS.same_params:  # same amount of compute
                fmap = slim.stack(x, multiscale_atrousconv, channel_sizes)
            else:  # same num of params
                fmap = slim.stack(x, multiscale_atrousconv,
                                  [n*c for c in channel_sizes[:-1]])
        else:
            fmap = slim.stack(x, slim.conv2d, [(k, (3, 3), (1, 1), 'SAME')
                                                for k in channel_sizes])

    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)


    fmap_summ = tf.summary.image('fmap', tf.expand_dims(tf.reduce_max(fmap, axis=3), axis=-1))
    logits = slim.fully_connected(tf.reduce_mean(fmap, axis=[1,2]), 10, activation_fn=None)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    loss_summary = tf.summary.scalar('loss', loss,)
    train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    p = tf.nn.softmax(logits)
    acc = accuracy(p, y)

    train_accuracy = tf.summary.scalar('train_acc', acc)
    train_im = tf.summary.image('train_im', x)
    train = tf.summary.merge([train_accuracy, train_im, fmap_summ])

    test_accuracy = tf.summary.scalar('test_acc', acc)
    test_im = tf.summary.image('test_im', x)
    test = tf.summary.merge([test_accuracy, test_im])


    n = np.sum([np.prod(var.get_shape().as_list())
               for var in tf.trainable_variables()])
    print('num of vars {}'.format(n))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        step = 0
        for e in range(FLAGS.epochs):
            for i, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
                if FLAGS.scale:
                    batch_ims = random_scale(batch_ims)

                L, _ = sess.run([loss, train_step],
                                {x: batch_ims, y: batch_labels})
                # print('\rloss: {}'.format(L), end='', flush=True)

                if step%500==0:
                    ids = np.random.randint(0, 5000, 50)
                    batch_test_ims = test_ims[ids, ...]
                    batch_test_labels = test_labels[ids]
                    if FLAGS.scale:
                        batch_test_ims = random_scale(batch_test_ims)
                    loss_summ, train_summ = sess.run([loss_summary, train],
                                               {x: batch_ims, y: batch_labels})
                    writer.add_summary(train_summ, step)
                    writer.add_summary(loss_summ, step)
                    test_summ = sess.run(test,
                                 {x: batch_test_ims,
                                  y: batch_test_labels})
                    writer.add_summary(test_summ, step)
                step += 1

if __name__ == '__main__':
    tf.app.run()
