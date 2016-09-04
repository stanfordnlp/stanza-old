import codecs
import six


def uprint(s):
    if six.PY2:
        print(s.encode('utf-8'))
    else:
        print(s)


def urepr(s):
    if six.PY2:
        return repr(s).decode('unicode_escape')
    else:
        return repr(s)


def uopen(filename, *args, **kwargs):
    return codecs.open(filename, *args, encoding='utf-8', **kwargs)
