"""
Dataset module for managing text datasets.
"""
__author__ = 'victor'
from collections import OrderedDict
import random
import numpy as np


class InvalidFieldsException(Exception):
    pass


class Dataset(object):
    """
    Generic Dataset object that encapsulates a list of instances.

    The dataset stores the instances in an ordered dictionary of fields.
    Each field maps to a list, the ith element of the list for field 'foo' corresponds to the attribute 'foo' for the ith instance in the dataset.

    The dataset object supports indexing, iterating, slicing (eg. for iterating over batches), shuffling,
    conversion to/from CONLL format, among others.

    Example:

    .. code-block:: python

        d = Dataset({'Name': ['Alice', 'Bob', 'Carol', 'David', 'Ellen'], 'SSN': [1, 23, 45, 56, 7890]})
        print(d)  # Dataset(Name, SSN)
        print(d[2])  # OrderedDict([('SSN', 45), ('Name', 'Carol')])
        print(d[1:3])  # OrderedDict([('SSN', [23, 45]), ('Name', ['Bob', 'Carol'])])

        for e in d:
            print(e)  # OrderedDict([('SSN', 1), ('Name', 'Alice')]) ...
    """

    def __init__(self, fields):
        """
        :param fields: An ordered dictionary in which a key is the name of an attribute and a value is a list of the values of the instances in the dataset.

        :return: A Dataset object
        """
        self.fields = OrderedDict(fields)
        length = None
        length_field = None
        for name, d in fields.items():
            if length is None:
                length = len(d)
                length_field = name
            else:
                if len(d) != length:
                    raise InvalidFieldsException('field {} has length {} but field {} has length {}'.format(length_field, length, name, len(d)))

    def __len__(self):
        """
        :return: The number of instances in the dataset.
        """
        if len(self.fields) == 0:
            return 0
        return len(self.fields.values()[0])

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ', '.join(self.fields.keys()))

    @classmethod
    def load_conll(cls, fname):
        """
        The CONLL file must have a tab delimited header, for example::

            # description   tags
            Alice
            Hello   t1
            my      t2
            name    t3
            is      t4
            alice   t5

            Bob
            I'm     t1
            bob     t2

        Here, the fields are `description` and `tags`. The first instance has the label `Alice` and the
        description `['Hello', 'my', 'name', 'is', 'alice']` and the tags `['t1', 't2', 't3', 't4', 't5']`.
        The second instance has the label `Bob` and the description `["I'm", 'bob']` and the tags `['t1', 't2']`.

        :param fname: The CONLL formatted file from which to load the dataset

        :return: loaded Dataset instance
        """
        def process_cache(cache, fields):
            cache = [l.split() for l in cache if l]
            if not cache:
                return None
            fields['label'].append(cache[0][0])
            instance = {k: [] for k in fields if k != 'label'}
            for l in cache[1:]:
                for i, k in enumerate(fields):
                    if k != 'label':
                        instance[k].append(None if l[i] == '-' else l[i])
            for k, v in instance.items():
                fields[k].append(v)

        cache = []

        with open(fname) as f:
            header = f.next().strip().split('\t')
            header[0] = header[0].lstrip('# ')
            fields = OrderedDict([(head, []) for head in header])
            fields['label'] = []
            for line in f:
                line = line.strip()
                if line:
                    cache.append(line)
                else:
                    # met empty line, process cache
                    process_cache(cache, fields)
                    cache = []
            if cache:
                process_cache(cache, fields)
        return cls(fields)

    def write_conll(self, fname):
        """
        Serializes the dataset in CONLL format to fname
        """
        if 'label' not in self.fields:
            raise InvalidFieldsException("dataset is not in CONLL format: missing label field")

        def instance_to_conll(inst):
            tab = [v for k, v in inst.items() if k != 'label']
            return '{}\n{}'.format(inst['label'], '\n'.join(['\t'.join(['-' if e is None else str(e) for e in row]) for row in zip(*tab)]))

        with open(fname, 'wb') as f:
            f.write('# {}'.format('\t'.join([k for k in self.fields if k != 'label'])))
            for i, d in enumerate(self):
                f.write('\n{}'.format(instance_to_conll(d)))
                if i != len(self) - 1:
                    f.write('\n')

    def convert(self, converters, in_place=False):
        """
        Applies transformations to the dataset.

        :param converters: A dictionary specifying the function to apply to each field. If a field is missing from the dictionary, then it will not be transformed.

        :param in_place: Whether to perform the transformation in place or create a new dataset instance

        :return: the transformed dataset instance
        """
        dataset = self if in_place else self.__class__(OrderedDict([(name, data[:]) for name, data in self.fields.items()]))
        for name, convert in converters.items():
            if name not in self.fields.keys():
                raise InvalidFieldsException('Converter specified for non-existent field {}'.format(name))
            for i, d in enumerate(dataset.fields[name]):
                dataset.fields[name][i] = convert(d)
        return dataset

    def shuffle(self):
        """
        Re-indexes the dataset in random order

        :return: the shuffled dataset instance
        """
        order = range(len(self))
        random.shuffle(order)
        for name, data in self.fields.items():
            reindexed = []
            for _, i in enumerate(order):
                reindexed.append(data[i])
            self.fields[name] = reindexed
        return self

    def __getitem__(self, item):
        """
        :param item: An integer index or a slice (eg. 2, 1:, 1:5)

        :return: an ordered dictionary of the instance(s) at index/indices `item`.
        """
        return OrderedDict([(name, data[item]) for name, data in self.fields.items()])

    def __setitem__(self, key, value):
        """
        :param key: An integer index or a slice (eg. 2, 1:, 1:5)

        :param value: Sets the instances at index/indices `key` to the instances(s) `value`
        """
        for name, data in self.fields.items():
            if name not in value:
                raise InvalidFieldsException('field {} is missing in input data: {}'.format(name, value))
            data[key] = value[name]

    def __iter__(self):
        """
        :return: A iterator over the instances in the dataset
        """
        for i in xrange(len(self)):
            yield self[i]

    def copy(self, keep_fields=None):
        """
        :param keep_fields: if specified, then only the given fields will be kept
        :return: A deep copy of the dataset (each instance is copied).
        """
        keep_fields = self.fields.keys() or keep_fields
        return self.__class__(OrderedDict([(name, data[:]) for name, data in self.fields.items() if name in keep_fields]))

    @classmethod
    def pad(cls, sequences, padding, pad_len=None):
        """
        Pads a list of sequences such that they form a matrix.

        :param sequences: a list of sequences of varying lengths.
        :param padding: the value of padded cells.
        :param pad_len: the length of the maximum padded sequence.
        """
        max_len = max([len(s) for s in sequences])
        pad_len = pad_len or max_len
        assert pad_len >= max_len, 'pad_len {} must be greater or equal to the longest sequence {}'.format(pad_len, max_len)
        for i, s in enumerate(sequences):
            sequences[i] = [padding] * (pad_len - len(s)) + s
        return np.array(sequences)
