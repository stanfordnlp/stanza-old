from itertools import islice, chain


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
