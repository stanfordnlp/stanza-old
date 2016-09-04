import tensorflow as tf

__author__ = 'kelvinguu'


def labels_to_onehots(labels, num_classes):
    """Convert a vector of integer class labels to a matrix of one-hot target vectors.

    :param labels: a vector of integer labels, 0 to num_classes. Has shape (batch_size,).
    :param num_classes: the total number of classes
    :return: has shape (batch_size, num_classes)
    """
    batch_size = labels.get_shape().as_list()[0]

    with tf.name_scope("one_hot"):
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        sparse_ptrs = tf.concat(1, [indices, labels], name="ptrs")
        onehots = tf.sparse_to_dense(sparse_ptrs, [batch_size, num_classes],
                                     1.0, 0.0)
        return onehots