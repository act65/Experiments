from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sonnet as snt

import numpy as np
import os

from utils import *

# from sklearn.utils import shuffle

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_integer('width', 32, 'width')
tf.app.flags.DEFINE_integer('rank', 4, 'rank')
tf.app.flags.DEFINE_integer('depth', 8, 'depth')
# TODO. allow for a soft transition between noise and not
FLAGS = tf.app.flags.FLAGS

class BasisLayer(snt.AbstractModule):
    def __init__(self, U, name='basis'):
        super(BasisLayer, self).__init__(name=name)
        self.U = U
        self.V_T = U.transpose()
        with self._enter_variable_scope():
            self.S = tf.get_variable(name='weights',
                                     shape=U.shape[0],
                                     dtype=tf.float32)
            self.bias = tf.get_variable(name='bias',
                                        shape=U.shape[1],
                                        dtype=tf.float32)

    def _build(self, inputs):
        outputs = self.U(self.S * self.V_T(inputs)) + self.bias
        # TODO want to add self normalising elus?
        return tf.nn.elu(outputs)

def selu(x):
    return tf.nn.elu(x)


class Basis(snt.AbstractModule):
    def __init__(self, shape, weights=None, name='basis'):
        super(Basis, self).__init__(name=name)
        self.shape = shape
        with self._enter_variable_scope():
            if weights is None:
                self.weights = tf.get_variable(name='weights', shape=shape, dtype=tf.float32)
            else:
                self.weights = weights

    def _build(self, x):
        return tf.matmul(x, self.weights)

    def transpose(self):
        return Basis(tuple(reversed(self.shape)), tf.transpose(self.weights))

def network(x, dim, rank, depth):
    # dont train the first layer
    first = snt.Sequential([snt.Linear(dim), tf.nn.elu])
    x = first(x)

    with tf.variable_scope('train_vars'):
        U = Basis((rank, dim))
        layers = [BasisLayer(U) for _ in range(depth)]
        net = snt.Sequential(layers)
        y = net(x)

        last = snt.Linear(10)
        return last(y)

def main(_):
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=False)
    ims = np.reshape(mnist.train.images, [-1, 784]).astype(np.float32)
    labels = np.reshape(mnist.train.labels, [-1]).astype(np.int64)
    # ims, labels = shuffle(ims, labels)

    test_ims = np.reshape(mnist.test.images, [-1, 784]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)

    x = tf.placeholder(shape=[FLAGS.batchsize, 784], dtype=tf.float32)
    T = tf.placeholder(shape=[FLAGS.batchsize], dtype=tf.int64)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step = global_step.assign_add(1)

    logits = network(x, FLAGS.width, FLAGS.rank, FLAGS.depth)

    print(x, T, logits)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=T)
    print(loss)
    trainable_vars = tf.get_collection('trainable_variables', scope='train_vars')
    train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss,
       var_list=trainable_vars)

    num_vars = np.sum([np.prod(v.get_shape().as_list())
                       for v in trainable_vars])
    print('number of variables: {}'.format(num_vars))

    with tf.name_scope('metrics'):
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        acc = tf.contrib.metrics.streaming_accuracy(preds, T,
                                  metrics_collections='METRICS',
                                  updates_collections='METRIC_UPDATES')

    loss_summary = tf.summary.scalar('supervised', tf.reduce_mean(loss))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.logdir+'-{}'.format(num_vars), sess.graph)
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
