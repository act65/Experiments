from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

import numpy as np
import os

from utils import *
from losses import get_loss_fn

from sklearn.utils import shuffle

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_integer('valid_epochs', 50, 'number of times through dataset for '
                            'validation training.')
tf.app.flags.DEFINE_integer('N_labels', 200, 'number of labels to train on')
tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_float('valid_lr', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_string('loss_fn', 'orth', 'loss function for pretraining')
FLAGS = tf.app.flags.FLAGS


def main(_):
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=False)
    ims = np.reshape(mnist.train.images, [-1, 28, 28, 1]).astype(np.float32)
    labels = np.reshape(mnist.train.labels, [-1]).astype(np.int64)
    ims, labels = shuffle(ims, labels)
    # TODO. this makes it harder to compare. unless we do multiple runs

    test_ims = np.reshape(mnist.test.images, [-1, 28, 28, 1]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)

    x = tf.placeholder(shape=[FLAGS.batchsize, 28, 28, 1], dtype=tf.float32)
    tf.add_to_collection('inputs', x)
    T = tf.placeholder(shape=[FLAGS.batchsize], dtype=tf.int64)
    tf.add_to_collection('targets', T)

    # set up
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step = global_step.assign_add(1)
    main_opt = tf.train.AdamOptimizer(FLAGS.lr)
    e2e_opt = tf.train.AdamOptimizer(FLAGS.valid_lr)
    classifier_opt = tf.train.AdamOptimizer(FLAGS.valid_lr)

    # build the model
    with tf.variable_scope('representation') as scope:
        hidden = encoder(x) # TODO hidden should be [batch, N] embeddings
        unsupervised_loss = tf.add_n([get_loss_fn(name, hidden)
                                      for name in FLAGS.loss_fn.split('-')])

    with tf.variable_scope('classifier') as scope:
        logits = classifier(tf.reduce_mean(hidden, [1,2]))
        discrim_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=T))

    with tf.variable_scope('optimisers') as scope:
        pretrain_step = main_opt.minimize(
              unsupervised_loss,
              var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='representation'))
        train_step = classifier_opt.minimize(discrim_loss,
              var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='classifier'))
        e2e_step = e2e_opt.minimize(discrim_loss)

    with tf.name_scope('metrics'):
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        acc = tf.contrib.metrics.streaming_accuracy(preds, T,
                                  metrics_collections='METRICS',
                                  updates_collections='METRIC_UPDATES')

    # summaries
    pretrain_summary = tf.summary.scalar('unsupervised', unsupervised_loss)
    discrim_summary = tf.summary.scalar('supervised', discrim_loss)
    loss_summaries = tf.summary.merge([pretrain_summary, discrim_summary])

################################################################################
    """
    Given that we are doing unsupervised pretraining for a discrimination task,
    it makes sense to validate our model on discrimination.
    """
    def freeze_pretrain(sess, writer, step):
        """
        Question: how useful is the pretrained representation for
        discrimination?
        Measure: validation accuracy.
        """
        # TODO. instead pick a subset of classes. or binary 1 class vs rest
        # try 10 labels?!

        ### Vanilla
        train_labels = labels; valid_labels = test_labels
        train_ims = ims; valid_ims = test_ims

        ### Train new classifier
        # TODO. what if we also want to validate on other tasks? such as;
        # ability to reconstruct data, MI with data, ???,
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='classifier')
        sess.run(tf.variables_initializer(variables))


        # a different subset every time we validate?!?
        # idx = np.random.randint(0, len(train_labels), FLAGS.N_labels)
        idx = range(FLAGS.N_labels)

        for e in range(FLAGS.valid_epochs):
            for i, batch_ims, batch_labels in batch(train_ims[idx],
                                                    train_labels[idx],
                                                    FLAGS.batchsize):
                L, _ = sess.run([discrim_loss, train_step],
                                {x: batch_ims, T: batch_labels})
            print('\rvalid: train step: {} loss: {:.5f}'.format(e, L), end='')
            add_summary(writer, e+FLAGS.valid_epochs*step//100, 'valid-train/freeze', L)
        validate(sess, writer, step, x, T, valid_ims, valid_labels, FLAGS.batchsize, name='freeze')

    def pretrained_endtoend(sess, writer, saver, step):
        """
        Question(s):
            - how close is the pretrained representation to the
            final learned representation (after training on labels).
            - how good the the pretrained init?
        Measure: iterations to ?, max accuracy?, l2 sitance between init and
        final weights?
        """
        ### Vanilla
        train_labels = labels; valid_labels = test_labels
        train_ims = ims; valid_ims = test_ims

        # save the variables before we fine tune them on labels
        saver.save(sess, FLAGS.logdir+'/', step)

        idx = range(FLAGS.N_labels)

        # TODO. need to makes use i am not overfitting here!
        # could use early stopping? but need separate valid-valid data
        for e in range(FLAGS.valid_epochs):
            for i, batch_ims, batch_labels in batch(train_ims[idx],
                                                    train_labels[idx],
                                                    FLAGS.batchsize):
                L, _ = sess.run([discrim_loss, e2e_step],
                                {x: batch_ims, T: batch_labels})
            print('\rvalid: train step: {} loss: {:.5f}'.format(e, L), end='')
            add_summary(writer, e+FLAGS.valid_epochs*step//100, 'valid-train/e2e', L)
        validate(sess, writer, step, x, T, valid_ims, valid_labels, FLAGS.batchsize, name='e2e')
        # restore original variables to continue pretrianing
        ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
        saver.restore(sess, ckpt)

    def semisupervised(sess, writer, step, batch_ims, batch_labels):
        """
        Question:
            - how can extra unlabelled data be used to help a small set of
            labels generalise?
            - how does adding labels into unsupervised training effect the
            representation learnt?
        """
        L, _ = sess.run([discrim_loss, e2e_step],
                              {x: batch_ims, T: batch_labels})

        validate(sess, writer, step, x, T, valid_ims, valid_labels, FLAGS.batchsize)

    def embed(sess, step):
        """
        Let's have a look at the hidden representations learned by our different
        methods. We need to run our model on a subset of data, collect the
        hidden representations and then save into a fake checkpoint.
        """
        # get embeddings from tensorflow
        X = []; H = []; L = []
        for i, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
            if i >= 10000//FLAGS.batchsize: break
            # print('\r embed step {}'.format(i), end='', flush=True)
            X.append(batch_ims)
            H.append(sess.run(hidden, feed_dict={x: batch_ims}))
            L.append(batch_labels.reshape(FLAGS.batchsize))

        save_embeddings(os.path.join(FLAGS.logdir, 'embedding'+str(step)),
                        np.vstack(H),
                        np.vstack(L).reshape(10000),
                        images=np.vstack(X))

    with tf.Session() as sess:
        run_options, run_metadata = profile()
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope='representation'))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for e in range(FLAGS.epochs):
            for _, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
                step, L, _ = sess.run([global_step, unsupervised_loss, pretrain_step],
                                      {x: batch_ims, T: batch_labels},
                                      options=run_options, run_metadata=run_metadata)
                print('\rtrain step: {} loss: {:.5f}'.format(step, L), end='')

                # semi-supervised learning
                # TODO. want a better way to sample labels
                idx = np.random.randint(0, FLAGS.N_labels, FLAGS.batchsize)
                _ = sess.run(e2e_step, {x: ims[idx], T: labels[idx]})
                # TODO. how does running the update together effect things?
                # TODO. what about elastic weight consolidation? treating
                # semi supervised learning as a type of transfer!?

                if step%20==0:
                    summ = sess.run(loss_summaries, {x: batch_ims, T: batch_labels})
                    writer.add_summary(summ, step)

                if step%100==0:
                    # pretrained_endtoend(sess, writer, saver, step)
                    # freeze_pretrain(sess, writer, step)
                    validate(sess, writer, step, x, T, test_ims, test_labels,
                             FLAGS.batchsize, name='super')

                if step == 30:
                    trace(run_metadata, FLAGS.logdir)

                if step%500 == 0:
                    var = tf.get_collection('random_vars')
                    sess.run(tf.variables_initializer(var))

                # if step%10000==1:
                #     embed(sess, step-1)

if __name__ == '__main__':
    tf.app.run()
