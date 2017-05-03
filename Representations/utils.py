import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.utils import shuffle
import os

def encoder(x):
    with tf.variable_scope('encoder'):
        channel_sizes = [(16, 2), (32, 2), (64, 2)]
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.orthogonal_initializer(),
                            biases_initializer=tf.constant_initializer(0.0)):

            fmap = slim.stack(x, slim.conv2d, [(k, (3, 3), (s, s), 'SAME')
                                            for k, s in channel_sizes])
            return tf.reduce_mean(fmap, axis=[1,2])

def classifier(x):
    with tf.variable_scope('classifier'):
        return slim.fully_connected(x, 10, activation_fn=None)


def batch(ims, labels, batchsize):
    ims, labels = shuffle(ims, labels)
    shape = ims.shape
    for i in range(len(labels)//batchsize):
        yield (i, ims[i*batchsize:(i+1)*batchsize, ...],
                  labels[i*batchsize:(i+1)*batchsize, ...])

def save_embeddings(logdir, name, sess, writer, config, embeddings, labels, images=None):
    """
    Args:
        embeddings: A numpy array of shape (10000, features) and type float32.
        labels: a numpy array of int32's. (10000,)
    """
    embed_var = tf.Variable(embeddings, name='embeddings'+name)
    sess.run(embed_var.initializer)

    saver = tf.train.Saver(var_list=[embed_var])
    os.makedirs(logdir, exist_ok=True)
    fname = saver.save(sess, os.path.join(logdir, name+'.ckpt'),
                       write_meta_graph=False)

    print('\nembedding saved to {}'.format(fname))


    embedding = config.embeddings.add()
    embedding.tensor_name = embed_var.name
    embedding.metadata_path = os.path.join(logdir, name+'metadata.tsv')
    if images:
        img_path, img_size = save_images(images, sess)
        embedding.sprite.image_path = img_path
        embedding.sprite.single_image_dim.extend(img_size)

    projector.visualize_embeddings(writer, config)

    # write labels.
    with open(os.path.join(logdir, name+'metadata.tsv'), 'w') as metadata_file:
        metadata_file.write('Name\tClass\n')
        for i, L in enumerate(labels):
            metadata_file.write('%06d\t%s\n' % (i, L))


def save_images(images, session):
    """sticks images together appropriately and saves them. Returns the path
    they are saved to and the width and height in pixels of the components."""
    # first step is to figure out the aspect ratio of the images and therefore
    # how to tile them
    print(images.shape)
    images = images.transpose([0, 2, 1, 3])
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
            if FLAGS.image_borders:
                img = _draw_border(img)
            column.append(img)
        all_pics.append(column.copy())
    all_pics = [np.concatenate(col, axis=1) for col in all_pics]
    all_pics = np.concatenate(all_pics, axis=0)
    img_tensor = 1.0 - tf.constant(all_pics)
    # stick it together a few times to get rgba with transparent background
    img_tensor = tf.concat(2, [img_tensor,
                               img_tensor,
                               img_tensor,
                               0.5*((1.0-img_tensor)**3) + 0.5])

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
    imname = os.path.join(FLAGS.logdir, 'saved_embeddings', 'images.png')
    # turn back to 8 bit
    img_tensor *= 255
    img_tensor = tf.saturate_cast(img_tensor, tf.uint8)
    encoded_imgs = session.run(
        tf.image.encode_png(img_tensor))
    print('saving images')
    with open(imname, 'wb') as fp:
        fp.write(encoded_imgs)

    return imname, img_size
