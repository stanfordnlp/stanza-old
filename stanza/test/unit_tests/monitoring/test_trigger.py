__author__ = 'victor'

from unittest import TestCase
from stanza.monitoring.trigger import ThresholdTrigger, SlopeThresholdTrigger, PatienceTrigger


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
        e = SlopeThresholdTrigger(min_thresh=-1, max_thresh=1, time=2)
        self.assertFalse(e(1))
        self.assertFalse(e(2))
        self.assertFalse(e(1.5))
        self.assertTrue(e(0.4))

        e = SlopeThresholdTrigger(min_thresh=-1, max_thresh=1, time=2)
        self.assertFalse(e(1))
        self.assertFalse(e(2))
        self.assertTrue(e(3.1))
