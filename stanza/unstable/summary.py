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


class SummaryWriter(object):
    def __init__(self, filename, tick=5.0, max_queue_len=100):
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
        # TODO: support floating-point tensors, 4-D tensors, grayscale
        if len(val.shape) != 3:
            raise ValueError('`log_image` value should be a 3-D tensor, instead got shape %s' %
                             (val.shape,))
        if val.shape[2] != 3:
            raise ValueError('Last dimension of `log_image` value should be 3 (RGB), '
                             'instead got shape %s' %
                             (val.shape,))
        fakefile = StringIO()
        png.Writer(size=val.shape[:2]).write(
            fakefile, val.reshape(val.shape[0], val.shape[1] * val.shape[2]))
        encoded = fakefile.getvalue()
        # https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto
        RGB = 3
        image = Summary.Image(height=val.shape[0], width=val.shape[1],
                              colorspace=RGB, encoded_image_string=encoded)
        summary = Summary(value=[Summary.Value(tag=tag, image=image)])
        self.add_event(step, summary)

    def log_scalar(self, step, tag, val):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(val))])
        self.add_event(step, summary)

    def log_histogram(self, step, tag, val):
        hist = Histogram()
        hist.add(val)
        summary = Summary(value=[Summary.Value(tag=tag, histo=hist.encode_to_proto())])
        self.add_event(step, summary)

    def add_event(self, step, summary):
        t = time.time()
        e = Event(wall_time=t, step=step, summary=summary)
        self.queue.append(e)
        if t - self.last_append >= self.tick or len(self.queue) >= self.max_queue_len:
            self.flush()
            self.last_append = t

    def flush(self):
        if self.queue:
            with open(self.filename, 'ab') as outfile:
                write_events(outfile, self.queue)
                del self.queue[:]


POS_BUCKETS = 1e-12 * 1.1 ** np.arange(0, 776.)
POS_BUCKETS[-1] = sys.float_info.max
DEFAULT_BUCKETS = np.array(list(reversed(-POS_BUCKETS)) + [0.0] + list(POS_BUCKETS))


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
    num: 3
    sum: -0.75
    sum_squares: 2.5625
    bucket_limit: -2.0
    bucket_limit: -1.0
    bucket_limit: 0.0
    bucket_limit: 1.0
    bucket: 0.0
    bucket: 1.0
    bucket: 0.0
    bucket: 2.0
    bucket: 0.0
    <BLANKLINE>
    '''
    def __init__(self, bucket_limits=None):
        if bucket_limits is None:
            bucket_limits = DEFAULT_BUCKETS
        self.bucket_limits = bucket_limits
        self.clear()

    def clear(self):
        self.min = self.bucket_limits[-1]
        self.max = self.bucket_limits[0]
        self.num = 0
        self.sum = 0.0
        self.sum_squares = 0.0
        self.buckets = np.zeros((len(self.bucket_limits) + 1,))

    def add(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        arr = arr.flatten()

        self.min = min(self.min, arr.min())
        self.max = max(self.max, arr.max())
        self.sum += arr.sum()
        self.num += len(arr)
        self.sum_squares += (arr ** 2).sum()

        indices = np.searchsorted(self.bucket_limits, arr)
        new_counts = np.bincount(indices, minlength=self.buckets.shape[0])
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
            if (count > 0.0 or i >= len(self.bucket_limits) or
                    self.buckets[i + 1] > 0.0):
                bucket_limits.append(float(end))
                buckets.append(float(count))
        buckets.append(float(self.buckets[-1]))

        p.bucket_limit.extend(bucket_limits)
        p.bucket.extend(buckets)
        return p


class SummaryReaderException(Exception):
    pass


def masked_crc(data):
    crc = crc32(data) & 0xffffffff
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def read_events(stream):
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
        print('%d bytes' % data_len)
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
    for event in events:
        data = event.SerializeToString()
        len_field = struct.pack('<Q', len(data))
        len_crc = struct.pack('<I', masked_crc(len_field))
        data_crc = struct.pack('<I', masked_crc(data))
        stream.write(len_field)
        stream.write(len_crc)
        stream.write(data)
        stream.write(data_crc)
