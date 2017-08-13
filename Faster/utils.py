from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# from sklearn.utils import shuffle

def net(x, width, depth):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
        h = slim.stack(x, slim.fully_connected, [width] * depth)
    return slim.fully_connected(h, 10, activation_fn=None)

################################################################################

def batch(ims, labels, batchsize):
    # ims, labels = shuffle(ims, labels)
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

################################################################################


def principle_engienvector(grad, var):
    var_shape = var.get_shape().as_list()
    s, u, v = tf.svd(grad)

    # need to transpose v
    v_shape = v.get_shape().as_list()
    if len(v_shape) == 2:
        v = tf.transpose(v, [1, 0])
    elif len(v_shape) == 3:
        v = tf.transpose(v, [0, 2, 1])

    # pick the strongest component
    v = v[0, ...]
    u = u[0, ...]

    if v.get_shape().as_list() == var_shape:
        return v, var
    elif u.get_shape().as_list() == var_shape:
        return u, var
    else:
        raise SystemError

def unaggregated_grads_and_vars(loss, var_list):
    unaggregated = list(zip(*[tf.gradients(l, var_list)
                              for l in tf.unstack(loss)]))
    stacked = [tf.stack(v) for v in unaggregated]
    return list(zip(stacked, var_list))
