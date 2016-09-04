from collections import deque

__author__ = 'victor, kelvinguu'
import numpy as np


class Trigger(object):
    """
    A generic Trigger object that performs some action on some event
    """
    pass


class StatefulTriggerMixin(object):
    """
    A mix-in denoting triggers with memory
    """

    def reset(self):
        """
        reset the Trigger to its initial state (eg. by clear its memory)
        """
        raise NotImplementedError()


class MetricTrigger(object):
    """
    An abstract class denoting triggers that are based on some metric
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ThresholdTrigger(MetricTrigger):
    """
    Triggers when the variable crosses the min threshold or the max threshold.
    """

    def __init__(self, min_threshold=-float('inf'), max_threshold=float('inf')):
        """
        :param min_threshold: if the variable crosses this threshold then the trigger returns True.
        :param max_threshold: if the variable crosses this threshold then the trigger returns True.
        """
        super(MetricTrigger, self).__init__()
        self.min = min_threshold
        self.max = max_threshold

    def __call__(self, new_value):
        """
        :return: whether the value exceeds the predefined thresholds.
        """
        return new_value > self.max or new_value < self.min


class PatienceTrigger(MetricTrigger, StatefulTriggerMixin):
    """
    Triggers when N time steps has elapsed since the best value
    for the variable was encountered. N is denoted by the patience parameter.
    """

    def __init__(self, patience):
        """
        :param patience: how many consecutive suboptimal values to tolerate before triggering.
        """
        super(PatienceTrigger, self).__init__()
        self.patience = patience
        self.best_so_far = -float('inf')
        self.time_since_best = 0

    def __call__(self, new_value):
        """
        :param new_value: value for this iteration.
        :return: True if `self.patience` consecutive suboptimal values have been seen.
        """
        if new_value > self.best_so_far:
            self.best_so_far = new_value
            self.time_since_best = 0
            return False
        self.time_since_best += 1
        return self.time_since_best > self.patience

    def reset(self):
        """
        reset the Trigger to its initial state (eg. by clear its memory)
        """
        self.best_so_far = -float('inf')
        self.time_since_best = 0


class SlopeTrigger(MetricTrigger, StatefulTriggerMixin):
    """
    Triggers when the slope of the values in the most recent time window
    falls within the specified range (inclusive).

    The slope is approximated with a least squares fit on the data points
    in the window.

    Data points passed to the slope trigger are assumed to each be one
    unit apart on the x axis.
    """
    def __init__(self, range, window_size=10):
        """
        :param range: a tuple of minimum and maximum range to tolerate.
        :param window_size: how many points to use to estimate the slope
        """
        self.range = range
        self.window_size = window_size
        self.vals = deque(maxlen=window_size)

    def __call__(self, new_value):
        """
        :param new_value: value for this time step
        :return: True if the value falls within the predefined range.
        """
        self.vals.append(new_value)

        # not enough points to robustly estimate slope
        if len(self.vals) < self.window_size:
            return False

        return self.range[0] <= self.slope() <= self.range[1]

    def slope(self):
        """
        :return: the esitmated slope for points in the current window
        """
        x = range(self.window_size)
        y = self.vals
        slope, bias = np.polyfit(x, y, 1)
        return slope

    def reset(self):
        """
        reset the Trigger to its initial state (eg. by clear its memory)
        """
        self.vals = deque(maxlen=self.window_size)
