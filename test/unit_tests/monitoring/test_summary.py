__author__ = 'wmonroe4'

import numpy as np
from unittest import TestCase
import stanza.monitoring.summary as summary
from stanza.research.mockfs import patcher


class TestLargeFloats(TestCase):
    '''
    Proto serialization breaks if floats exceed the maximum of a float64.
    Make sure summary.py converts these to inf to avoid crashes.
    '''
    def test_large_hist(self):
        fs = patcher('stanza.monitoring.summary', '/test')
        open = fs.start()

        writer = summary.SummaryWriter('/test/large_hist.tfevents')
        writer.log_histogram(1, 'bighist', np.array(1.0e39))
        writer.flush()
        with open('/test/large_hist.tfevents', 'r') as infile:
            events = list(summary.read_events(infile))

        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0].summary.value), 1)
        self.assertTrue(events[0].summary.value[0].HasField('histo'),
                        events[0].summary.value[0])

        fs.stop()

    def test_large_scalar(self):
        fs = patcher('stanza.monitoring.summary', '/test')
        open = fs.start()

        writer = summary.SummaryWriter('/test/large_scalar.tfevents')
        writer.log_scalar(1, 'bigvalue', 1.0e39)
        writer.flush()
        with open('/test/large_scalar.tfevents', 'r') as infile:
            events = list(summary.read_events(infile))

        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0].summary.value), 1)
        self.assertTrue(np.isinf(events[0].summary.value[0].simple_value))

        fs.stop()
