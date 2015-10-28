__author__ = 'victor'

from unittest import TestCase
from stanza.monitoring.trigger import ThresholdEarlyStopping, SlopeThresholdEarlyStopping, PatienceEarlyStopping


class TestEarlyStopping(TestCase):

    def test_threshold(self):
        e = ThresholdEarlyStopping(min_threshold=-10, max_threshold=2)
        for val in xrange(-10, 3):
            self.assertFalse(e.should_stop(val))
        self.assertTrue(e.should_stop(-10.1))
        self.assertTrue(e.should_stop(2.1))

    def test_patience(self):
        e = PatienceEarlyStopping(patience=3)
        self.assertFalse(e.should_stop(10))
        self.assertFalse(e.should_stop(9))
        self.assertFalse(e.should_stop(8))
        self.assertFalse(e.should_stop(11))
        self.assertFalse(e.should_stop(10))
        self.assertFalse(e.should_stop(1))
        self.assertFalse(e.should_stop(10))
        self.assertTrue(e.should_stop(10))

    def test_slope_threshold(self):
        e = SlopeThresholdEarlyStopping(min_thresh=-1, max_thresh=1, time=2)
        self.assertFalse(e.should_stop(1))
        self.assertFalse(e.should_stop(2))
        self.assertFalse(e.should_stop(1.5))
        self.assertTrue(e.should_stop(0.4))

        e = SlopeThresholdEarlyStopping(min_thresh=-1, max_thresh=1, time=2)
        self.assertFalse(e.should_stop(1))
        self.assertFalse(e.should_stop(2))
        self.assertTrue(e.should_stop(3.1))
