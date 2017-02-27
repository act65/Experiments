from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate.')
FLAGS = tf.app.flags.FLAGS


# TODO. a couple of ways to do this.
# - sample layer weights from a larger set of parameters
# - use the same parameters, just shuffle them at each stage.

# does this sort of idea relate to cecho state nets?


def shuffle_var(var):
    """
    Args:
        var (tf.Tensor): any tensor

    Return:
        (tf.Tensor): a randomly shuffled tensor with values from var.
    """
    shape = var.get_shape()
    ids = range(np.prod(shape.as_list()))
    np.random.shuffle(ids)
    return tf.reshape(tf.gather(tf.reshape(var, [-1]), ids), shape)


def batch(ims, labels, batchsize):
    ims, labels = shuffle(ims, labels)
    shape = ims.shape
    for i in range(len(labels)//batchsize):
        yield (i, ims[i*batchsize:(i+1)*batchsize, ...],
               labels[i*batchsize:(i+1)*batchsize, ...])


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

    # channel_sizes = [16, 16, 16, 16]
    channel_sizes = [64, 32]
    n = 4

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
        # TODO need BN?
        if FLAGS.shared:
            if FLAGS.shuffled:

        else:
            fmap = slim.stack(x, slim.conv2d, [(k, (3, 3), (1, 1), 'SAME')
                                                for k in channel_sizes])


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
                print('\rloss: {}'.format(L), end='', flush=True)

                if step%500==0:
                    ids = np.random.randint(0, 5000, 50)
                    batch_test_ims = test_ims[ids, ...]
                    batch_test_labels = test_labels[ids]


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
