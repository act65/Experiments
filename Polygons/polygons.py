from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np
import cv2

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.app.flags.DEFINE_bool('atrous', False, 'whether to use atrous conv')
tf.app.flags.DEFINE_bool('scale', True, 'whether to randomly scale data')
tf.app.flags.DEFINE_bool('same_params', True, 'whether to use same n of params '
                         'vs same amount of compute.')
FLAGS = tf.app.flags.FLAGS

def get_polys(ims):
    polys = []
    for im in ims:
        image, contours, hierarchy = cv2.findContours(
                                    (ims[0]*255).astype(np.uint8),
                                    cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        polys.append(contours)
    return polys


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
    polys = get_polys(ims)

    test_ims = np.reshape(mnist.test.images, [-1, 28, 28, 1]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)
    test_polys = get_polys(test_ims)

    x = tf.placeholder(shape=[50, 28, 28, 1], dtype=tf.float32)
    y = tf.placeholder(shape=[50, None], dtype=tf.int64)
    # TODO. variable sized...

    channel_sizes = [32, 32, 32]

    p = act(x)
    # TODO. need to augment act to collect outputs rather than averaging?!

    fmap_summ = tf.summary.image('fmap', tf.expand_dims(tf.reduce_max(fmap, axis=3), axis=-1))
    # TODO would like to plot created polys...
    logits = slim.fully_connected(tf.reduce_mean(fmap, axis=[1,2]), 10, activation_fn=None)

    loss = tf.reduce_mean()
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


                L, _ = sess.run([loss, train_step],
                                {x: batch_ims, y: batch_labels})
                # print('\rloss: {}'.format(L), end='', flush=True)

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
