'''
>>> fs = patcher('stanza.monitoring.summary', '/test'); open = fs.start()
... # ^ for doctest; ignore

A nearly pure-Python module for logging output to a file in TensorBoard's
events format. Supports scalars, RGB images and histograms.

>>> writer = SummaryWriter('/test/values.tfevents')
>>> writer.log_scalar(1, 'universe', 42)
>>> writer.flush()
>>> with open('/test/values.tfevents', 'r') as infile:
...     for event in read_events(infile):
...         print(event)  # doctest: +ELLIPSIS
wall_time: ...
step: 1
summary {
  value {
    tag: "universe"
    simple_value: 42.0
  }
}
<BLANKLINE>

This module requires a very small subset of TensorFlow to be available for
importing, consisting of the following compiled protobuf definitions:

    tensorflow/core/
        framework/
            attr_value_pb2.py
            function_pb2.py
            graph_pb2.py
            op_def_pb2.py
            summary_pb2.py
            tensor_pb2.py
            tensor_shape_pb2.py
            types_pb2.py
        util/
            event_pb2.py

It also requires a couple of other easy-to-install Python modules:

    pip install -U pypng 'Protobuf>=3.0.0b2'

After an event file is written (it should have 'tfevents' somewhere in its
name), the file can be read by TensorBoard by running

    tensorboard --logdir="`pwd`"

from the parent directory of the directory containing the events file.

>>> fs.stop()
... # ^ for doctest; ignore
'''
__author__ = 'wmonroe4'

import atexit
import numpy as np
import png
import struct
import sys
import time
from itertools import izip
from StringIO import StringIO

from tensorflow.core.util.event_pb2 import Event
from tensorflow.core.framework.summary_pb2 import Summary, HistogramProto

from .crc32c import crc as crc32
from ..research.mockfs import patcher  # NOQA: for doctest

open = open  # for doctest


class SummaryWriter(object):
    def __init__(self, filename, tick=5.0, max_queue_len=100):
        '''
        :param str filename: The path of the events file to be written.
            The file is truncated during construction of the `SummaryWriter`
            object.
        :param float tick: The number of seconds to elapse in between
            automatically writing queued events out to the file. A write
            can be forced manually with a call to `flush()`.
        :param max_queue_len: The maximum number of events to keep queued
            before the queue is flushed. If more than this number of events
            accumulate in the queue, they will be flushed even if `tick`
            seconds have not elapsed.

        Note that event writing is performed synchronously; unlike the
        TensorFlow SummaryWriter, this module is not run in a separate
        thread or process.
        '''
        self.filename = filename

        self.tick = tick
        self.max_queue_len = max_queue_len

        self.last_append = time.time()
        self.queue = []

        # Truncate the file to start
        with open(filename, 'wb'):
            pass

        atexit.register(SummaryWriter.flush, self)

    def log_image(self, step, tag, val):
        '''
        Write an image event.

        :param int step: Time step (x-axis in TensorBoard graphs)
        :param str tag: Label for this value
        :param numpy.ndarray val: Image in RGB format with values from
            0 to 255; a 3-D array with index order (row, column, channel).
            `val.shape[-1] == 3`
        '''
        # TODO: support floating-point tensors, 4-D tensors, grayscale
        if len(val.shape) != 3:
            raise ValueError('`log_image` value should be a 3-D tensor, instead got shape %s' %
                             (val.shape,))
        if val.shape[2] != 3:
            raise ValueError('Last dimension of `log_image` value should be 3 (RGB), '
                             'instead got shape %s' %
                             (val.shape,))
        fakefile = StringIO()
        png.Writer(size=(val.shape[1], val.shape[0])).write(
            fakefile, val.reshape(val.shape[0], val.shape[1] * val.shape[2]))
        encoded = fakefile.getvalue()
        # https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto
        RGB = 3
        image = Summary.Image(height=val.shape[0], width=val.shape[1],
                              colorspace=RGB, encoded_image_string=encoded)
        summary = Summary(value=[Summary.Value(tag=tag, image=image)])
        self._add_event(step, summary)

    def log_scalar(self, step, tag, val):
        '''
        Write a scalar event.

        :param int step: Time step (x-axis in TensorBoard graphs)
        :param str tag: Label for this value
        :param float val: Scalar to graph at this time step (y-axis)
        '''
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(np.float32(val)))])
        self._add_event(step, summary)

    def log_histogram(self, step, tag, val):
        '''
        Write a histogram event.

        :param int step: Time step (x-axis in TensorBoard graphs)
        :param str tag: Label for this value
        :param numpy.ndarray val: Arbitrary-dimensional array containing
            values to be aggregated in the resulting histogram.
        '''
        hist = Histogram()
        hist.add(val)
        summary = Summary(value=[Summary.Value(tag=tag, histo=hist.encode_to_proto())])
        self._add_event(step, summary)

    def _add_event(self, step, summary):
        t = time.time()
        e = Event(wall_time=t, step=step, summary=summary)
        self.queue.append(e)
        if t - self.last_append >= self.tick or len(self.queue) >= self.max_queue_len:
            self.flush()
            self.last_append = t

    def flush(self):
        '''
        Force all queued events to be written to the events file.
        The queue will automatically be flushed at regular time intervals,
        when it grows too large, and at program exit (with the usual caveats
        of `atexit`: this won't happen if the program is killed with a
        signal or `os._exit()`).
        '''
        if self.queue:
            with open(self.filename, 'ab') as outfile:
                write_events(outfile, self.queue)
                del self.queue[:]


_default_buckets = None


def default_buckets():
    global _default_buckets
    if _default_buckets is None:
        positive_buckets = 1e-12 * 1.1 ** np.arange(0, 776.)
        positive_buckets[-1] = sys.float_info.max
        _default_buckets = np.array(list(reversed(-positive_buckets)) + [0.0] +
                                    list(positive_buckets))
    return _default_buckets


class Histogram(object):
    '''
    Stores statistics about the values of an array as counts of values
    falling into buckets on a logarithmic scale.

    Ported from the TensorFlow C++ class:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/histogram/histogram.cc

    >>> h = Histogram([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> h.add([-1.5, 0.5, 0.25])
    >>> print(str(h.encode_to_proto()))
    min: -1.5
    max: 0.5
    num: 3.0
    sum: -0.75
    sum_squares: 2.5625
    bucket_limit: -2.0
    bucket_limit: -1.0
    bucket_limit: 0.0
    bucket_limit: 1.0
    bucket_limit: 2.0
    bucket: 0.0
    bucket: 1.0
    bucket: 0.0
    bucket: 2.0
    bucket: 0.0
    <BLANKLINE>
    '''
    def __init__(self, bucket_limits=None):
        if bucket_limits is None:
            bucket_limits = default_buckets()
        self.bucket_limits = bucket_limits
        self.clear()

    def clear(self):
        self.min = self.bucket_limits[-1]
        self.max = self.bucket_limits[0]
        self.num = 0
        self.sum = 0.0
        self.sum_squares = 0.0
        self.buckets = np.zeros((len(self.bucket_limits),))

    def add(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        arr = arr.flatten()

        self.min = min(self.min, arr.min())
        self.max = max(self.max, arr.max())
        self.sum += arr.sum()
        self.num += len(arr)
        self.sum_squares += (arr ** 2).sum()

        indices = np.searchsorted(self.bucket_limits, arr, side='right')
        new_counts = np.bincount(indices, minlength=self.buckets.shape[0])
        if new_counts.shape[0] > self.buckets.shape[0]:
            # This should only happen with nans and extremely large values
            assert new_counts.shape[0] == self.buckets.shape[0] + 1, new_counts.shape
            new_counts = new_counts[:self.buckets.shape[0]]
        self.buckets += new_counts

    def encode_to_proto(self):
        p = HistogramProto()
        p.min = float(self.min)
        p.max = float(self.max)
        p.num = float(self.num)
        p.sum = float(self.sum)
        p.sum_squares = float(self.sum_squares)

        bucket_limits = []
        buckets = []
        for i, (end, count) in enumerate(izip(self.bucket_limits, self.buckets)):
            if (i == len(self.bucket_limits) - 1 or
                    count > 0.0 or self.buckets[i + 1] > 0.0):
                bucket_limits.append(float(end))
                buckets.append(float(count))

        p.bucket_limit.extend(bucket_limits)
        p.bucket.extend(buckets)
        return p


class SummaryReaderException(Exception):
    pass


def masked_crc(data):
    crc = crc32(data) & 0xffffffff
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def read_events(stream):
    '''
    Read and return as a generator a sequence of Event protos from
    file-like object `stream`.
    '''
    header_size = struct.calcsize('<QI')
    len_size = struct.calcsize('<Q')
    footer_size = struct.calcsize('<I')

    while True:
        header = stream.read(header_size)
        if len(header) == 0:
            break
        elif len(header) < header_size:
            raise SummaryReaderException('unexpected EOF (expected a %d-byte header, '
                                         'got %d bytes)' % (header_size, len(header)))
        data_len, len_crc = struct.unpack('<QI', header)
        len_crc_actual = masked_crc(header[:len_size])
        if len_crc_actual != len_crc:
            raise SummaryReaderException('incorrect length CRC (%d != %d)' %
                                         (len_crc_actual, len_crc))

        data = stream.read(data_len)
        if len(data) < data_len:
            raise SummaryReaderException('unexpected EOF (expected %d bytes, got %d)' %
                                         (data_len, len(data)))
        yield Event.FromString(data)

        footer = stream.read(footer_size)
        if len(footer) < footer_size:
            raise SummaryReaderException('unexpected EOF (expected a %d-byte footer, '
                                         'got %d bytes)' % (footer_size, len(footer)))
        data_crc, = struct.unpack('<I', footer)
        data_crc_actual = masked_crc(data)
        if data_crc_actual != data_crc:
            raise SummaryReaderException('incorrect data CRC (%d != %d)' %
                                         (data_crc_actual, data_crc))


def write_events(stream, events):
    '''
    Write a sequence of Event protos to file-like object `stream`.
    '''
    for event in events:
        data = event.SerializeToString()
        len_field = struct.pack('<Q', len(data))
        len_crc = struct.pack('<I', masked_crc(len_field))
        data_crc = struct.pack('<I', masked_crc(data))
        stream.write(len_field)
        stream.write(len_crc)
        stream.write(data)
        stream.write(data_crc)


__all__ = [
    'SummaryWriter',
    'read_events',
    'write_events',
]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: summary.py [summary_file.tfevents]')
        sys.exit(1)
    with open(sys.argv[1], 'rb') as infile:
        for event in read_events(infile):
            print(event)
