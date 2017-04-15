import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np

# https://arxiv.org/abs/1702.05538

tf.app.flags.DEFINE_string('logdir', '/tmp/test', 'location for saved embeedings')
tf.app.flags.DEFINE_string('datadir', '/tmp/mnist', 'location for data')
tf.app.flags.DEFINE_integer('batchsize', 50, 'batch size.')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of times through dataset.')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate.')
FLAGS = tf.app.flags.FLAGS


def batch(ims, labels, batchsize):
    ims, labels = shuffle(ims, labels)
    shape = ims.shape
    for i in range(len(labels)//batchsize):
        yield (i, ims[i*batchsize:(i+1)*batchsize, ...],
               labels[i*batchsize:(i+1)*batchsize, ...])


def accuracy(p, y):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p, axis=1), y), tf.float32))


def encoder(x, shapes=[10]):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
                        return slim.stack(x, slim.fully_connected, shapes)

def decoder(x, shapes=[10]):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.orthogonal_initializer(),
                        biases_initializer=tf.constant_initializer(0.0)):
        return slim.stack(x, slim.fully_connected, shapes)


def principle_pertubation():
    # a simple/cheaper version of this could just be the average of the
    # variance of batches?
    pass

def neighborhood_pertubation():
    # if same label then ...
    pass

# relatedly. how to come up with cheap adversarial examples? what about adding
# adversarial pertubations into the feature space?


def main(_):
    print('Get data')
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=False)
    ims = np.reshape(mnist.train.images, [-1, 28, 28, 1]).astype(np.float32)
    labels = np.reshape(mnist.train.labels, [-1]).astype(np.int64)

    test_ims = np.reshape(mnist.test.images, [-1, 28, 28, 1]).astype(np.float32)
    test_labels = np.reshape(mnist.test.labels, [-1]).astype(np.int64)

    x = tf.placeholder(shape=[50, 28*28], dtype=tf.float32)
    y = tf.placeholder(shape=[50], dtype=tf.int64)
    h = encoder(x)
    hidden_summ = tf.summary.histogram('hidden', h)

    # TODO what are we comparing this against?
    # - denoising AEs. add gaussian noise to x
    # - contractive AEs. min dy/dx (??)
    # ?!?!?


    if FLAGS.principle:
        # TODO. where should this be done is a classification setting?
        # the final layer, the logits?
        S, U, V = tf.svd(h)
        z = tf.random_normal(shape=h.get_shape(), dtype=tf.float32, stddev=1.0)
        h += tf.mul(z, U)
        # add noise along principle component. outtper prod with noise
        # add noise scaled by principle component
        # would also like a version of this that explicity interpolates/extrapolates.

    if FLAGS.neighbors:
        for x in tf.unstack(h):
            for y in tf.unstack(h):
                if x != y:
                    c = x-y

    if FLAGS.


    logits = h
    # logits = decoder(h)


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    loss_summary = tf.summary.scalar('loss', loss,)
    train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    p = tf.nn.softmax(logits)
    acc = accuracy(p, y)

    train_accuracy = tf.summary.scalar('train_acc', acc)
    train_im = tf.summary.image('train_im', x)
    train = tf.summary.merge([train_accuracy, train_im, hidden_summ])

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
                if FLAGS.scale:
                    batch_ims = random_scale(batch_ims)

                L, _ = sess.run([loss, train_step],
                                {x: batch_ims, y: batch_labels})
                print('\rloss: {}'.format(L), end='', flush=True)

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
