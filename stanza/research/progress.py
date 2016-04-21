"""A module for periodically displaying progress on a hierarchy of tasks
and estimating time to completion.

>>> import progress, datetime
>>> progress.set_resolution(datetime.datetime.resolution)  # show all messages, don't sample
>>> progress.start_task('Repetition', 2)
>>> for rep in range(2):  # doctest: +ELLIPSIS
...     progress.progress(rep)
...     progress.start_task('Example', 3)
...     for ex in range(3):
...         progress.progress(ex)
...     progress.end_task()
...
Repetition 0 of 2 (~0% done, ETA unknown on ...)
Repetition 0 of 2, Example 0 of 3 (~0% done, ETA unknown on ...)
Repetition 0 of 2, Example 1 of 3 (~17% done, ETA ...)
Repetition 0 of 2, Example 2 of 3 (~33% done, ETA ...)
Repetition 0 of 2, Example 3 of 3 (~50% done, ETA ...)
Repetition 1 of 2 (~50% done, ETA ...)
Repetition 1 of 2, Example 0 of 3 (~50% done, ETA ...)
Repetition 1 of 2, Example 1 of 3 (~67% done, ETA ...)
Repetition 1 of 2, Example 2 of 3 (~83% done, ETA ...)
Repetition 1 of 2, Example 3 of 3 (~100% done, ETA ...)
>>> progress.end_task()  # doctest: +ELLIPSIS
Repetition 2 of 2 (~100% done, ETA ...)
"""

import datetime
import doctest
from collections import namedtuple


class ProgressMonitor(object):
    def __init__(self, resolution=datetime.datetime.resolution):
        self.task_stack = []
        self.last_report = datetime.datetime.min
        self.resolution = resolution
        self.start_time = datetime.datetime.now()

    def start_task(self, name, size):
        if len(self.task_stack) == 0:
            self.start_time = datetime.datetime.now()
        self.task_stack.append(Task(name, size, 0))

    def progress(self, p):
        self.task_stack[-1] = self.task_stack[-1]._replace(progress=p)
        self.progress_report()

    def end_task(self):
        self.progress(self.task_stack[-1].size)
        self.task_stack.pop()

    def progress_report(self):
        now = datetime.datetime.now()
        if (len(self.task_stack) > 1 or self.task_stack[0] > 0) and \
                now - self.last_report < self.resolution:
            return

        stack_printout = ', '.join('%s %s of %s' % (t.name, t.progress, t.size)
                                   for t in self.task_stack)

        frac_done = self.fraction_done()
        if frac_done == 0.0:
            now_str = now.strftime('%c')
            eta_str = 'unknown on %s' % now_str
        else:
            elapsed = (now - self.start_time)
            estimated_length = elapsed.total_seconds() / frac_done
            eta = self.start_time + datetime.timedelta(seconds=estimated_length)
            eta_str = eta.strftime('%c')

        print '%s (~%d%% done, ETA %s)' % (stack_printout,
                                           round(frac_done * 100.0),
                                           eta_str)
        self.last_report = datetime.datetime.now()

    def fraction_done(self, start=0.0, finish=1.0, stack=None):
        if stack is None:
            stack = self.task_stack

        if len(stack) == 0:
            return start
        else:
            top_fraction = stack[0].progress * 1.0 / stack[0].size
            next_top_fraction = (stack[0].progress + 1.0) / stack[0].size
            inner_start = start + top_fraction * (finish - start)
            inner_finish = start + next_top_fraction * (finish - start)
            return self.fraction_done(inner_start, inner_finish, stack[1:])


Task = namedtuple('Task', ('name', 'size', 'progress'))

_global_t = ProgressMonitor(resolution=datetime.timedelta(minutes=1))


def start_task(name, size):
    _global_t.start_task(name, size)


def progress(p):
    _global_t.progress(p)


def end_task():
    _global_t.end_task()


def set_resolution(res):
    _global_t.resolution = res


if __name__ == '__main__':
    doctest.testmod()
