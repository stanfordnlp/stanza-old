import six
__author__ = ['kelvinguu']


def to_unicode(s):
    """Return the object as unicode (only matters for Python 2.x).

    If s is already Unicode, return s as is.
    Otherwise, assume that s is UTF-8 encoded, and convert to Unicode.

    :param (basestring) s: a str, unicode or other basestring object
    :return (unicode): the object as unicode
    """
    if not isinstance(s, six.string_types):
        raise ValueError("{} must be str or unicode.".format(s))
    if not isinstance(s, six.text_type):
        s = six.text_type(s, 'utf-8')
    return s
