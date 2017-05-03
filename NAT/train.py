from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.utils import shuffle
import numpy as np
import os

from losses import nat, siamese
from utils import *

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate.')
FLAGS = tf.app.flags.FLAGS

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

    # parameterised functions
    hidden = encoder(x)
    logits = classifier(hidden)

    # losses
    unsupervised_loss = nat(hidden)
    discrim_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=T)

    # optimisation steps
    pretrain_step = opt.minimize(unsupervised_loss, global_step=global_step,
          var_list=tf.get_collection('TRAINABLE_VARIABLES', scope='encoder'))
    train_step = opt.minimize(discrim_loss,
          var_list=tf.get_collection('TRAINABLE_VARIABLES', scope='classifier'))

    # metrics
    preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
    acc = tf.contrib.metrics.streaming_accuracy(preds, T,
                                  metrics_collections='METRICS',
                                  updates_collections='METRIC_UPDATES')

    # summaries
    loss_summary = tf.summary.merge([tf.summary.scalar(loss.name, loss)
                                     for loss in [unsupervised_loss, discrim_loss]])
    accuracy_summary = tf.summary.scalar('acc', acc)

    def train(sess, writer, batch_ims):
        if step%5 == 0:
            loss_summ, step, L, _ = sess.run([loss_summary, global_step, loss,
                                              pretrain_step], {x: batch_ims})
            writer.add_summary(loss_summ, step)
        else:
            step, L, _ = sess.run([global_step, loss, pretrain_step], {x: batch_ims})
        print('\rloss: {:.3f}'.format(L), end='', flush=True)
        return step

    def validate(sess, writer, global_step):
        """
        Given that we are doing unsupervised pretraining for a discrimination task,
        it makes sense to validate our model on discrimination.
        """
        ### Train new classifier
        variables = tf.get_collection('TRAINABLE_VARIABLES', scope='classifier')
        sess.run(tf.variables_initializer(variables))

        # train classifier (only) on a labelled subset of the data
        # TODO a subset
        for i, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
            sess.run(train_step, {x: batch_ims, T: batch_labels})

        ### Validate classifier
        metrics = tf.get_collection('METRICS')
        updates = tf.get_collection('METRIC_UPDATES')
        variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='valid')
        sess.run(tf.variables_initializer(variables))

        # eval and aggregate
        for _, batch_ims, batch_labels in batch(test_ims, test_labels, FLAGS.batchsize):
            sess.run(updates)
        values = sess.run(metrics, {x: batch_ims, T: batch_labels})

        # write
        for k, v in zip(metrics, values):
            summ = tf.Summary(value=[tf.Summary.Value(tag='valid/' + k.name,
                                                      simple_value=float(v))])
            writer.add_summary(summ, global_step)


    # def embed(sess):
    #     x, h, l = get_embeddings()
    #     save_embeddings(sess, h, l, images=x)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for e in range(FLAGS.epochs):
            for _, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
                step = train(sess, writer, batch_ims)
                if step%50==0:
                    validate(sess, writer)

if __name__ == '__main__':
    tf.app.run()
