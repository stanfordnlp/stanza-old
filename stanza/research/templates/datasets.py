from collections import namedtuple

from stanza.research.instance import Instance
from stanza.research.rng import get_rng


rng = get_rng()


# TODO: replace these silly datasets with real data
def foobar_train():
    return [Instance(input='foo', output='bar') for _ in range(1000)]


def foobar_dev():
    return [Instance(input='foo', output='bar') for _ in range(100)]


def foobar_test():
    return [Instance(input='foo', output='bar') for _ in range(100)]


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'foobar_dev': DataSource(foobar_train, foobar_dev),
    'foobar_test': DataSource(foobar_train, foobar_test),
}
