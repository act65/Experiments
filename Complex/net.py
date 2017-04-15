from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim  # try out sonnet instead?
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.utils import shuffle
import numpy as np
import os

from utils import *

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
    print('Get data')
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=False)
    ims = np.reshape(mnist.train.images, [-1, 784]).astype(np.float32)
    labels = np.reshape(mnist.train.labels, [-1]).astype(np.int64)

    test_ims = np.reshape(mnist.test.images, [-1, 784]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)

    x = tf.placeholder(shape=[50, 784], dtype=tf.complex64)
    l = tf.placeholder(shape=[50], dtype=tf.int64)
    L = tf.one_hot(l, 10, 0.0, 1.0)

    channel_sizes = [50, 10]

    with slim.arg_scope([complex_fc],
                        activation_fn=None,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
        logits = slim.stack(x, complex_fc, channel_sizes)
    y = tf.nn.softmax(tf.abs(logits))
    loss = tf.reduce_mean(-L*tf.log(y)-(1-L)*tf.log(1-y))
    loss_summary = tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    step_summary = tf.summary.scalar('global_step', global_step)


    opt = tf.train.AdamOptimizer(FLAGS.lr)
    logdir = os.path.join(FLAGS.logdir, ''.join([str(c) for c in channel_sizes]))
    train_step = opt.minimize(loss, global_step=global_step)

    p = tf.nn.softmax(tf.abs(logits))
    acc = accuracy(p, l)

    train_accuracy = tf.summary.scalar('train_acc', acc)
    train_im = None #tf.summary.image('train_im', x)
    train = tf.summary.merge([train_accuracy])  #  train_im, fmap_summ

    test_accuracy = tf.summary.scalar('test_acc', acc)
    test_im = None #tf.summary.image('test_im', x)
    test = tf.summary.merge([test_accuracy])  # test_im


    n = np.sum([np.prod(var.get_shape().as_list())
               for var in tf.trainable_variables()])
    print('num of vars {}'.format(n))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        step = 0
        for e in range(FLAGS.epochs):
            for i, batch_ims, batch_labels in batch(ims, labels, FLAGS.batchsize):
                L, _ = sess.run([loss, train_step],
                                {x: batch_ims.astype(np.complex64), l: batch_labels},
                                options=run_options, run_metadata=run_metadata)
                print('\rloss: {:.3f}'.format(L), end='', flush=True)

                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(os.path.join(logdir, 'timeline.json'), 'w') as f:
                    f.write(ctf)

                if step%500==0:
                    # validate and save summaries
                    ids = np.random.randint(0, 5000, 50)
                    batch_test_ims = test_ims[ids, ...]
                    batch_test_labels = test_labels[ids]
                    loss_summ, train_summ = sess.run( [loss_summary, train],
                                        {x: batch_ims.astype(np.complex64), l: batch_labels})
                    writer.add_summary(train_summ, step)
                    writer.add_summary(loss_summ, step)
                    test_summ = sess.run(test,
                                 {x: batch_test_ims.astype(np.complex64),
                                  l: batch_test_labels})
                    writer.add_summary(test_summ, step)
                step += 1

if __name__ == '__main__':
    tf.app.run()
