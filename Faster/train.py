from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

import numpy as np
import os

from utils import *

# from sklearn.utils import shuffle

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 8, 'batch size.')
tf.app.flags.DEFINE_integer('depth', 10, 'depth')
tf.app.flags.DEFINE_integer('width', 64, 'width')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_bool('random_labels', False, 'whether to permute the labels')
# TODO. allow for a soft transition between noise and not
FLAGS = tf.app.flags.FLAGS

def main(_):
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=False)
    ims = np.reshape(mnist.train.images, [-1, 784]).astype(np.float32)
    labels = np.reshape(mnist.train.labels, [-1]).astype(np.int64)
    # ims, labels = shuffle(ims, labels)

    if FLAGS.random_labels:
        print('randomising the labels')
        labels = np.random.permutation(labels)
    print(ims.shape, labels.shape)

    test_ims = np.reshape(mnist.test.images, [-1, 784]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)

    x = tf.placeholder(shape=[FLAGS.batchsize, 784], dtype=tf.float32)
    T = tf.placeholder(shape=[FLAGS.batchsize], dtype=tf.int64)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step = global_step.assign_add(1)

    logits = net(x, FLAGS.width, FLAGS.depth)

    print(x, T, logits)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=T)
    print(loss)
    opt = tf.train.AdamOptimizer(FLAGS.lr)
    gnvs = unaggregated_grads_and_vars(loss, tf.trainable_variables())
    gnvs = [principle_engienvector(g, v) for g, v in gnvs]
    # gnvs = [tf.clip(g, ?, ?), v for g, v in gnvs]

    train_step = opt.apply_gradients(gnvs)

    num_vars = np.sum([np.prod(v.get_shape().as_list())
                       for v in tf.trainable_variables()])
    print('number of variables: {}'.format(num_vars))

    with tf.name_scope('metrics'):
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        acc = tf.contrib.metrics.streaming_accuracy(preds, T,
                                  metrics_collections='METRICS',
                                  updates_collections='METRIC_UPDATES')

    loss_summary = tf.summary.scalar('supervised', tf.reduce_mean(loss))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for e in range(FLAGS.epochs):
            for _, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
                step, L, _ = sess.run([global_step, loss, train_step],
                                      {x: batch_ims, T: batch_labels})
                print('\rtrain step: {} loss: {:.5f}'.format(step, np.mean(L)), end='')

                if step%50==0:
                    summ = sess.run(loss_summary, {x: batch_ims, T: batch_labels})
                    writer.add_summary(summ, step)
                if step%200==0:
                    validate(sess, writer, step, x, T, test_ims, test_labels,
                             FLAGS.batchsize, name='supervised')


if __name__ == '__main__':
    tf.app.run()
