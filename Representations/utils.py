import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.utils import shuffle
import os

import numpy as np

################################################################################
# TODO. would be nicer if these were sonnet modules?!
def classifier(x):
    with tf.variable_scope('fc'):
        x = tf.reduce_mean(x, axis=[1,2])
        return slim.fully_connected(x, 10, activation_fn=None)


def encoder(x):
    with tf.variable_scope('convnet'):
        channel_sizes = [(16, 2), (32, 2)]
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.orthogonal_initializer(),
                            biases_initializer=tf.constant_initializer(0.0)):

            return slim.stack(x, slim.conv2d, [(k, (3, 3), (s, s), 'SAME')
                                            for k, s in channel_sizes])

def decoder(x):
    with tf.variable_scope('deconvnet'):
        channel_sizes = [(16, 2), (32, 2)]
        with slim.arg_scope([deconv2d],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.orthogonal_initializer(),
                            biases_initializer=tf.constant_initializer(0.0)):

            fmap = slim.stack(x, deconv2d, [(k, (3, 3), s, 'SAME')
                                            for k, s in channel_sizes])
        outputs = slim.conv2d(fmap, 1, (1, 1), (1,1), 'SAME',
                              activation_fn=None,
                              weights_initializer=tf.orthogonal_initializer(),
                              biases_initializer=tf.constant_initializer(0.0))
        return outputs


@slim.add_arg_scope
def deconv2d(x, num_outputs=None, kernel_size=None, stride=None, padding='SAME',
           shape=None, activation_fn=None, scope='', method='upsample',
           resize_method=tf.image.ResizeMethod.BILINEAR, weights_initializer=None,
           biases_initializer=None, normalizer_fn=None, normalizer_params=None,
           **kwargs):
    """
    Upsample-conv method for deconvolution.
    See http://distill.pub/2016/deconv-checkerboard/ for more details.

    Args:
        x (tensor): An image, shape `[batch, height, width, k]`
        channels (int): the number of output feature maps
        shape: a tuple of ints. (new_width, new_height)

    Returns:
        - An image. [batch, shape[0], shape[1], channels]
    """
    with tf.variable_scope(scope or 'deconv'):
        if shape is not None:  # if provided a target shape
            x = tf.image.resize_images(
                x, shape[1:3] if len(shape) == 4 else shape, method=resize_method)
        else:  # else just use the stride
            shape = x.get_shape().as_list()
            # TODO. this is not robust to input shapes not to the power of 2
            h = shape[1]*stride #(shape[1]-kernel_size[0])*stride + 2
            w = shape[2]*stride
            x = tf.image.resize_images(
                x, [h, w], method=resize_method)
            return slim.conv2d(x, num_outputs=num_outputs, kernel_size=kernel_size, stride=[1, 1],
                               padding=padding, scope='deconv'+scope,
                               activation_fn=activation_fn, **kwargs)


################################################################################

def batch(ims, labels, batchsize):
    ims, labels = shuffle(ims, labels)
    shape = ims.shape
    for i in range(len(labels)//batchsize):
        yield (i, ims[i*batchsize:(i+1)*batchsize, ...],
                  labels[i*batchsize:(i+1)*batchsize, ...])


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
# def binarise(x, n=4):
#     y = x.copy()
#     y[y!=n] = 0
#     y[y==n] = 1
#     return y
# train_labels = binarise(labels)
# valid_labels = binarise(test_labels)
# train_ims = ims; valid_ims = test_ims

################################################################################


def validate(sess, writer, step, x, T, valid_ims, valid_labels, batchsize):
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
        summ = tf.Summary(value=[tf.Summary.Value(tag='valid/' + k.name,
                                                  simple_value=float(v))])
        writer.add_summary(summ, step)


################################################################################

def save_embeddings(logdir, embeddings, labels, images=None):
    """
    Args:
        embeddings: A numpy array of shape (10000, features) and type float32.
        labels: a numpy array of int32's. (10000,)
    """
    with tf.Graph().as_default() as g:
        sess = tf.Session()
        embed_var = tf.Variable(embeddings, name='embeddings')
        sess.run(embed_var.initializer)

        saver = tf.train.Saver(var_list=[embed_var])
        writer = tf.summary.FileWriter(logdir, sess.graph)
        os.makedirs(logdir, exist_ok=True)
        fname = saver.save(sess, os.path.join(logdir, 'model.ckpt'),
                           write_meta_graph=False)

        print('\nembedding saved to {}'.format(fname))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embed_var.name
        embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
        if images is not None:
            img_path, img_size = save_images(logdir, images, sess)
            embedding.sprite.image_path = img_path
            embedding.sprite.single_image_dim.extend(img_size)

        projector.visualize_embeddings(writer, config)

    # write labels.
    with open(os.path.join(logdir, 'metadata.tsv'), 'w') as metadata_file:
        metadata_file.write('Name\tClass\n')
        for i, L in enumerate(labels):
            metadata_file.write('%06d\t%s\n' % (i, L))


def save_images(logdir, images, sess):
    """sticks images together appropriately and saves them. Returns the path
    they are saved to and the width and height in pixels of the components."""
    # first step is to figure out the aspect ratio of the images and therefore
    # how to tile them
    # images = images.transpose([0, 2, 1, 3])
    # for our purposes aspect is width/height
    aspect_ratio = images.shape[1] / images.shape[2]
    total = images.shape[0]
    sqrt = np.sqrt(total) + 1
    num_across = int(sqrt * np.sqrt(aspect_ratio))
    num_down = int(sqrt / np.sqrt(aspect_ratio))

    # pull them out into an appropriate list
    all_pics = []
    img_index = 0
    for row in range(num_down):
        column = []
        for col in range(num_across):
            if img_index < total:
                img = images[img_index, ...]
            else:
                img = np.zeros_like(images[0, ...])
            img_index += 1
            column.append(img)
        all_pics.append(column.copy())
    all_pics = [np.concatenate(col, axis=1) for col in all_pics]
    all_pics = np.concatenate(all_pics, axis=0)
    img_tensor = 1.0 - tf.constant(all_pics)
    # stick it together a few times to get rgba with transparent background
    img_tensor = tf.concat([img_tensor,
                            img_tensor,
                            img_tensor,
                            0.5*((1.0-img_tensor)**3) + 0.5], axis=2)

    if all_pics.shape[0] > 8192 or all_pics.shape[1] > 8192:
        img_tensor = tf.expand_dims(img_tensor, 0)
        # get the factors we need to compress each axis
        factors = [8192 / dim for dim in all_pics.shape[:-1]]
        img_size = [int(dim * factor)
                    for dim, factor in zip(images.shape[1:3], factors)]
        new_size = [img_size[0] * num_down, img_size[1] * num_across]
        img_tensor = tf.image.resize_images(
            img_tensor, new_size, align_corners=True)
        img_tensor = tf.squeeze(img_tensor, 0)
    else:
        img_size = images.shape[1:3]
    # this seems necessary?
    img_size = [img_size[1], img_size[0]]
    imname = os.path.join(logdir, 'images.png')
    # turn back to 8 bit
    img_tensor *= 255
    img_tensor = tf.saturate_cast(img_tensor, tf.uint8)
    encoded_imgs = sess.run(
        tf.image.encode_png(img_tensor))
    with open(imname, 'wb') as fp:
        fp.write(encoded_imgs)

    return imname, img_size
