from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from sklearn.utils import shuffle

def wide_net(x):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
        h = slim.fully_connected(x, 1000)
    return slim.fully_connected(h, 10, activation_fn=None)


def deep_net(x):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
        h = slim.stack(x, slim.fully_connected, [200] * 17)
    return slim.fully_connected(h, 10, activation_fn=None)

################################################################################

def batch(ims, labels, batchsize):
    ims, labels = shuffle(ims, labels)
    shape = ims.shape
    for i in range(len(labels)//batchsize):
        yield (i, ims[i*batchsize:(i+1)*batchsize, ...],
                  labels[i*batchsize:(i+1)*batchsize, ...])


def validate(sess, writer, step, x, T, valid_ims, valid_labels, batchsize, name=''):
    ### Validate classifier
    metrics = tf.get_collection('METRICS')
    updates = tf.get_collection('METRIC_UPDATES')
    variables = tf.get_collection('LOCAL_VARIABLES', scope='metrics')
    sess.run(tf.variables_initializer(variables))

    # eval and aggregate
    for _, batch_ims, batch_labels in batch(valid_ims, valid_labels, batchsize):
        sess.run(updates, {x: batch_ims, T: batch_labels})
    values = sess.run(metrics, {x: batch_ims, T: batch_labels})

    # write summary
    for k, v in zip(metrics, values):
        add_summary(writer, step, 'valid/'+name, float(v))

def add_summary(writer, step, name, val):
    summ = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=val)])
    writer.add_summary(summ, step)

def get_loss_fn(name, logits):
    return
