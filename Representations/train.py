from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

import numpy as np
import os

from losses import nat, siamese, orth
from utils import *

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_integer('valid_epochs', 50, 'number of times through dataset for '
                            'validation training.')
tf.app.flags.DEFINE_integer('N_labels', 200, 'number of labels to train on')
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_bool('pretrain', True, 'whether to pretrain')
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
    main_opt = tf.train.AdamOptimizer(FLAGS.lr)
    classifier_opt = tf.train.AdamOptimizer(0.1)

    with tf.variable_scope('representation') as scope:
        hidden = encoder(x)
        unsupervised_loss = orth(hidden, 1.0)
        pretrain_step = main_opt.minimize(unsupervised_loss, global_step=global_step,
              var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))

    with tf.variable_scope('classifier') as scope:
        logits = classifier(hidden)
        discrim_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=T))
        train_step = classifier_opt.minimize(discrim_loss,
              var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))

    with tf.name_scope('metrics'):
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        acc = tf.contrib.metrics.streaming_accuracy(preds, T,
                                  metrics_collections='METRICS',
                                  updates_collections='METRIC_UPDATES')

    # summaries
    train_summary = tf.summary.scalar('pretraining', unsupervised_loss)
    valid_summary = tf.summary.scalar('training', discrim_loss)


    def validate(sess, writer, global_step):
        """
        Given that we are doing unsupervised pretraining for a discrimination task,
        it makes sense to validate our model on discrimination.
        """
        # TODO. instead pick a subset of classes. or binary 1 class vs rest


        ### Choose a subset of classes to train and validate on
        # def subset(x, y):
        #     zeros = x == 0
        #     ones = x == 1
        #     twos = x == 2
        #     return (np.concatenate([x[zeros], x[ones], x[twos]]),
        #             np.concatenate([y[zeros], y[ones], y[twos]]))
        # train_labels, train_ims = subset(labels, ims)
        # valid_labels, valid_ims = subset(test_labels, test_ims)

        ### Choose a single class for binary classification
        def binarise(x, n=4):
            y = x.copy()
            y[y!=n] = 0
            y[y==n] = 1
            return y
        train_labels = binarise(labels)
        valid_labels = binarise(test_labels)
        train_ims = ims; valid_ims = test_ims

        ### Vanilla
        # train_labels = labels; valid_labels = test_labels
        # train_ims = ims; valid_ims = test_ims

        ### Train new classifier
        # TODO. what if we also want to validate on other tasks? such as;
        # ability to reconstruct data, MI with data, ???,
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        sess.run(tf.variables_initializer(variables))
        # TODO what if I want to jointly train for validation but then reset to
        # previos values? could use a checkpoint to restore from after training.
        # NOTE. this is closer to what we actually want.

        # a different subset every time we validate?!?
        idx = np.random.randint(0, len(train_labels), FLAGS.N_labels)

        for e in range(FLAGS.valid_epochs):
            for i, batch_ims, batch_labels in batch(train_ims[idx],
                                                    train_labels[idx],
                                                    FLAGS.batchsize):
                L, _ = sess.run([discrim_loss, train_step],
                                {x: batch_ims, T: batch_labels})
            summ = sess.run(valid_summary, {x: batch_ims, T: batch_labels})
            writer.add_summary(summ, e+FLAGS.valid_epochs*step//100)

        ### Validate classifier
        metrics = tf.get_collection('METRICS')
        updates = tf.get_collection('METRIC_UPDATES')
        variables = tf.get_collection('LOCAL_VARIABLES', scope='metrics')
        sess.run(tf.variables_initializer(variables))

        # eval and aggregate
        for _, batch_ims, batch_labels in batch(valid_ims, valid_labels, FLAGS.batchsize):
            sess.run(updates, {x: batch_ims, T: batch_labels})
        values = sess.run(metrics, {x: batch_ims, T: batch_labels})

        # write summary
        for k, v in zip(metrics, values):
            summ = tf.Summary(value=[tf.Summary.Value(tag='valid/' + k.name,
                                                      simple_value=float(v))])
            writer.add_summary(summ, global_step)

    def embed(sess, step):
        """
        Let's have a look at the hidden representations learned by our different
        methods. We need to run our model on a subset of data, collect the
        hidden representations and then save into a fake checkpoint.
        """
        X = []; H = []; L = []
        for i, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
            if i >= 200: break
            # print('\r embed step {}'.format(i), end='', flush=True)
            X.append(batch_ims)
            H.append(sess.run(hidden, feed_dict={x: batch_ims}))
            L.append(batch_labels.reshape(50))
        save_embeddings(os.path.join(FLAGS.logdir, 'embedding'+str(step)),
                        np.vstack(H),
                        np.vstack(L).reshape(10000),
                        images=np.vstack(X))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for e in range(FLAGS.epochs):
            for _, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
                step, L, _ = sess.run([global_step, unsupervised_loss,
                                       pretrain_step if FLAGS.pretrain else None],
                                      {x: batch_ims})
                print('\rtrain step: {} loss: {:.5f}'.format(step, L), end='', flush=True)

                if step%20==0:
                    summ = sess.run(train_summary, {x: batch_ims})
                    writer.add_summary(summ, step)

                if step%100==0:
                    validate(sess, writer, step)

                if step%1000==1:
                    embed(sess, step-1)

if __name__ == '__main__':
    tf.app.run()
