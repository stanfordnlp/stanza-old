__author__ = 'victor, kelvinguu'

from unittest import TestCase
from stanza.monitoring.trigger import ThresholdTrigger, SlopeTrigger, PatienceTrigger


class TestEarlyStopping(TestCase):

    def test_threshold(self):
        e = ThresholdTrigger(min_threshold=-10, max_threshold=2)
        for val in xrange(-10, 3):
            self.assertFalse(e(val))
        self.assertTrue(e(-10.1))
        self.assertTrue(e(2.1))

    def test_patience(self):
        e = PatienceTrigger(patience=3)
        self.assertFalse(e(10))
        self.assertFalse(e(9))
        self.assertFalse(e(8))
        self.assertFalse(e(11))
        self.assertFalse(e(10))
        self.assertFalse(e(1))
        self.assertFalse(e(10))
        self.assertTrue(e(10))

    def test_slope_threshold(self):
        e = SlopeTrigger(range=(-1, 1), window_size=2)
        self.assertFalse(e(1))  # not enough points
        self.assertTrue(e(2))  # slope = 1
        self.assertFalse(e(4))  # slope 2 > 1
        self.assertFalse(e(2))  # slope -2 < -1
        self.assertTrue(e(2))  # slope = 0
