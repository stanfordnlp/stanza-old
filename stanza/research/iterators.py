from itertools import islice, imap, chain


def iter_batches(iterable, batch_size):
    '''
    Given a sequence or iterable, yield batches from that iterable until it
    runs out. Note that this function returns a generator, and also each
    batch will be a generator.

    :param iterable: The sequence or iterable to split into batches
    :param int batch_size: The number of elements of `iterable` to iterate over
        in each batch

    >>> batches = iter_batches('abcdefghijkl', batch_size=5)
    >>> list(next(batches))
    ['a', 'b', 'c', 'd', 'e']
    >>> list(next(batches))
    ['f', 'g', 'h', 'i', 'j']
    >>> list(next(batches))
    ['k', 'l']
    >>> list(next(batches))
    Traceback (most recent call last):
        ...
    StopIteration

    Warning: It is important to iterate completely over each batch before
    requesting the next, or batch sizes will be truncated to 1. For example,
    making a list of all batches before asking for the contents of each
    will not work:

    >>> batches = list(iter_batches('abcdefghijkl', batch_size=5))
    >>> len(batches)
    12
    >>> list(batches[0])
    ['a']

    However, making a list of each individual batch as it is received will
    produce expected behavior (as shown in the first example).
    '''
    # http://stackoverflow.com/a/8290514/4481448
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, batch_size)
        yield chain([batchiter.next()], batchiter)


def gen_batches(iterable, batch_size):
    '''
    Returns a generator object that yields batches from `iterable`.
    See `iter_batches` for more details and caveats.

    Note that `iter_batches` returns an iterator, which never supports `len()`,
    `gen_batches` returns an iterable which supports `len()` if and only if
    `iterable` does. This *may* be an iterator, but could be a `SizedGenerator`
    object. To obtain an iterator (for example, to use the `next()` function),
    call `iter()` on this iterable.

    >>> batches = gen_batches('abcdefghijkl', batch_size=5)
    >>> len(batches)
    3
    >>> for batch in batches:
    ...     print(list(batch))
    ['a', 'b', 'c', 'd', 'e']
    ['f', 'g', 'h', 'i', 'j']
    ['k', 'l']
    '''
    def batches_thunk():
        return iter_batches(iterable, batch_size)

    try:
        length = len(iterable)
    except TypeError:
        return batches_thunk()

    num_batches = (length - 1) // batch_size + 1
    return SizedGenerator(batches_thunk, length=num_batches)


class SizedGenerator(object):
    '''
    A class that wraps a generator to support len().

    Usage:

    >>> func = lambda: (x ** 2 for x in range(5))
    >>> gen = SizedGenerator(func, length=5)
    >>> len(gen)
    5
    >>> list(gen)
    [0, 1, 4, 9, 16]

    `length=None` can be passed to have the length be inferred (a O(n)
    running time operation):

    >>> func = lambda: (x ** 2 for x in range(5))
    >>> gen = SizedGenerator(func, length=None)
    >>> len(gen)
    5
    >>> list(gen)
    [0, 1, 4, 9, 16]

    Caller is responsible for assuring that provided length, if any,
    matches the actual length of the sequence:

    >>> func = lambda: (x ** 2 for x in range(8))
    >>> gen = SizedGenerator(func, length=10)
    >>> len(gen)
    10
    >>> len(list(gen))
    8

    Note that this class has the following caveats:

    * `func` must be a callable that can be called with no arguments.
    * If length=None is passed to the constructor, the sequence yielded by `func()`
      will be enumerated once during the construction of this object. This means
      `func()` must yield sequences of the same length when called multiple times.
      Also, assuming you plan to enumerate the sequence again to use it, this can
      double the time spent going through the sequence!

    The last requirement is because in general it is not possible to predict the
    length of a generator sequence, so we actually observe the output for one
    run-through and assume the length will stay the same on a second run-through.
    '''
    def __init__(self, func, length):
        self.func = func
        if length is None:
            length = sum(1 for _ in func())
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.func())


def sized_imap(func, iterable, strict=False):
    '''
    Return an iterable whose elements are the result of applying the callable `func`
    to each element of `iterable`. If `iterable` has a `len()`, then the iterable returned
    by this function will have the same `len()`. Otherwise calling `len()` on the
    returned iterable will raise `TypeError`.

    :param func: The function to apply to each element of `iterable`.
    :param iterable: An iterable whose objects will be mapped.
    :param bool strict: If `True` and `iterable` does not support `len()`, raise an exception
        immediately instead of returning an iterable that does not support `len()`.
    '''
    try:
        length = len(iterable)
    except TypeError:
        if strict:
            raise
        else:
            return imap(func, iterable)
    return SizedGenerator(lambda: imap(func, iterable), length=length)
