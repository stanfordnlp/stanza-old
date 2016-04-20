import contextlib
import mock
import os
import StringIO


def yields(thing):
    yield thing


class MockOpen(object):
    def __init__(self, test_dir):
        self.files = {}
        self.old_open = open
        self.test_dir = test_dir

    def __call__(self, filename, mode, *args, **kwargs):
        if filename.startswith(self.test_dir):
            if filename not in self.files or mode in ('w', 'w+'):
                self.files[filename] = StringIO.StringIO()
            fakefile = self.files[filename]
            if mode in ('r', 'r+'):
                fakefile.seek(0)
            else:
                fakefile.seek(0, os.SEEK_END)
            return contextlib.contextmanager(yields)(fakefile)
        else:
            return self.old_open(filename, *args, **kwargs)


def patcher(module, test_dir):
    mo = MockOpen(test_dir)
    return mock.patch(module + '.open', mo)
