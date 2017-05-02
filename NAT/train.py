from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.utils import shuffle
import numpy as np
import os

from losses import nat, siamese

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate.')
FLAGS = tf.app.flags.FLAGS


def batch(ims, labels, batchsize):
    ims, labels = shuffle(ims, labels)
    shape = ims.shape
    for i in range(len(labels)//batchsize):
        yield (i, ims[i*batchsize:(i+1)*batchsize, ...],
               labels[i*batchsize:(i+1)*batchsize, ...])

def accuracy(p, y):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p, axis=1), y), tf.float32))

def main(_):
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=False)
    ims = np.reshape(mnist.train.images, [-1, 28, 28, 1]).astype(np.float32)
    labels = np.reshape(mnist.train.labels, [-1]).astype(np.int64)

    test_ims = np.reshape(mnist.test.images, [-1, 28, 28, 1]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)

    x = tf.placeholder(shape=[50, 28, 28, 1], dtype=tf.float32)
    T = tf.placeholder(shape=[50], dtype=tf.int64)

    # set up
    global_step = tf.Variable(0, name='global_step', trainable=False)
    step_summary = tf.summary.scalar('global_step', global_step)
    opt = tf.train.AdamOptimizer(FLAGS.lr)

    def encoder(x):
        with tf.variable_scope('encoder'):
            channel_sizes = [(16, 2), (32, 2), (64, 2)]
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.orthogonal_initializer(),
                                biases_initializer=tf.constant_initializer(0.0)):

                return slim.stack(x, slim.conv2d, [(k, (3, 3), (s, s), 'SAME')
                                                for k, s in channel_sizes])

    def classifier(x):
        with tf.variable_scope('classifier'):
            return slim.fully_connected(tf.reduce_mean(x, axis=[1,2]), 10,
                                 activation_fn=None)

    # parameterised functions
    fmap = encoder(x)
    logits = classifier(fmap)

    # losses
    unsupervised_loss = nat(fmap)
    discrim_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=T)

    # optimisation steps
    pretrain_step = opt.minimize(unsupervised_loss, global_step=global_step,
                              var_list=encoder_vars)
    train_step = opt.minimize(discrim_loss, var_list=discrim_vars)

    # metrics
    p = tf.nn.softmax(logits)
    acc = accuracy(p, T)  # TODO. switch to streaming accuracy

    # summaries
    loss_summary = tf.summary.merge([tf.summary.scalar(loss.name, loss)
                                     for loss in [unsupervised_loss, discrim_loss]])
    accuracy_summary = tf.summary.scalar('acc', acc)

    def train(sess, writer, batch_ims):
        step, L, _ = sess.run([global_step, loss, pretrain_step], {x: batch_ims})
        print('\rloss: {:.3f}'.format(L), end='', flush=True)
        writer.add_summary(loss_summ, step)
        return step

    def validate(sess, writer):
        """
        Given that we are doing unsupervised pretraining for a discrimination task,
        it makes sense to validate our model on discrimination.
        """
        # reset
        inits = [var.initializer for var in tf.get_collection('TRAINABLE_VARIABLES',
                                                              scope='classifier')]
        sess.run(inits)  # also need to reset metrics

        # train classifier (only) on a labelled subset of the data
        for i, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
            sess.run(train_step, {x: batch_ims, T: batch_labels})

        for i, batch_ims, batch_labels in batch(test_ims, test_labels,
                                                FLAGS.batchsize):
            sess.run(acc_op)

        writer.add_summary(sess.run(acc), step)


    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for e in range(FLAGS.epochs):
            for _, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
                step = train(sess, writer, batch_ims)

                if step%50==0:
                    # validate and save summaries
                    validate(sess, writer)

if __name__ == '__main__':
    tf.app.run()
