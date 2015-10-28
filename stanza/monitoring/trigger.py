__author__ = 'victor'
import numpy as np


class Trigger(object):

    def should_stop(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class EarlyStopping(Trigger):

    def should_stop(self, new_value):
        raise NotImplementedError()


class ThresholdEarlyStopping(EarlyStopping):

    def __init__(self, min_threshold=-float('inf'), max_threshold=float('inf')):
        super(ThresholdEarlyStopping, self).__init__()
        self.min = min_threshold
        self.max = max_threshold

    def should_stop(self, new_value):
        return new_value > self.max or new_value < self.min


class PatienceEarlyStopping(EarlyStopping):

    def __init__(self, patience):
        super(PatienceEarlyStopping, self).__init__()
        self.patience = patience
        self.best_so_far = -float('inf')
        self.time_since_best = 0

    def should_stop(self, new_value):
        if new_value > self.best_so_far:
            self.best_so_far = new_value
            self.time_since_best = 0
            return False
        self.time_since_best += 1
        return self.time_since_best > self.patience

    def reset(self):
        self.best_so_far = -float('inf')
        self.time_since_best = 0


class SlopeThresholdEarlyStopping(EarlyStopping):

    def __init__(self, min_thresh=-float('inf'), max_thresh=float('inf'), time=5):
        super(SlopeThresholdEarlyStopping, self).__init__()
        self.min = min_thresh
        self.max = max_thresh
        self.points_seen = 0
        self.time = time
        self.x = np.array(range(time))
        self.y = np.zeros(time)
        # this is for framing the slope interpolation as least squares regression
        self.A = np.vstack([self.x, np.ones(len(self.x))]).T

    def should_stop(self, new_value):
        # shift time by 1
        self.y[:-1] = self.y[1:]
        self.y[-1] = new_value

        if self.points_seen < self.time-1:
            self.points_seen += 1
            return False # not enough data points to interpolate

        # x is time, y is values. We interpolate the m and c in y = mx + c
        m, c = np.linalg.lstsq(self.A, self.y)[0]
        return m < self.min or m > self.max

    def reset(self):
        self.points_seen = 0
        self.y.fill(0)
