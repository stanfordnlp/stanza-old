try:
    import tensorflow as tf
    tf.InteractiveSession
except (ImportError, AttributeError):
    tf_present = False
else:
    tf_present = True

import numpy as np
from stanza.ml.tensorflow_utils import labels_to_onehots
from unittest import TestCase, skipIf

__author__ = 'kelvinguu'


@skipIf(not tf_present, 'Tensorflow is not present')
class TestTensorFlowUtils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = tf.InteractiveSession()

    def test_labels_to_onehots(self):
        labels_list = [0, 1, 2, 3, 1]

        labels = tf.constant(labels_list, dtype=tf.int32)
        onehots = labels_to_onehots(labels, 5)
        result = onehots.eval()

        correct = np.zeros((5, 5))
        for i in range(5):
            correct[i, labels_list[i]] = 1

        # self.assertTrue(False)
        self.assertTrue(np.all(result == correct))
