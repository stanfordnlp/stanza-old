__author__ = 'victor'

from unittest import TestCase
from stanza.text.dataset import Dataset, InvalidFieldsException
from tempfile import NamedTemporaryFile
import random
from collections import OrderedDict
import os


class TestDataset(TestCase):

    CONLL = """# description\ttags
Alice
Hello\tt1
my\tt2
name\tt3
is\tt4
alice\tt5

Bob
I'm\tt1
bob\tt2"""

    CONLL_MOCK = OrderedDict([
        ('description', [
            ['Hello', 'my', 'name', 'is', 'alice'],
            ["I'm", 'bob'],
            ]),
        ('tags', [
            ['t1', 't2', 't3', 't4', 't5'],
            ['t1', 't2'],
        ]),
        ('label', ['Alice', 'Bob']),
    ])

    MOCK = OrderedDict([('Name', ['Alice', 'Bob', 'Carol']), ('SSN', ['123', None, '7890'])])

    def setUp(self):
        random.seed(1)
        self.mock = Dataset(OrderedDict([(name, d[:]) for name, d in self.MOCK.items()]))
        self.conll = Dataset(OrderedDict([(name, d[:]) for name, d in self.CONLL_MOCK.items()]))

    def test_init(self):
        self.assertRaises(InvalidFieldsException, lambda: Dataset({'name': ['alice', 'bob'], 'ssn': ['1']}))

    def test_length(self):
        self.assertEqual(0, len(Dataset({})))
        self.assertEqual(2, len(Dataset({'name': ['foo', 'bar']})))

    def test_load_conll(self):
        with NamedTemporaryFile() as f:
            f.write(self.CONLL)
            f.flush()
            d = Dataset.load_conll(f.name)
            self.assertDictEqual(self.CONLL_MOCK, d.fields)

    def test_write_conll(self):
        f = NamedTemporaryFile(delete=False)
        f.close()
        d = Dataset(self.CONLL_MOCK)
        d.write_conll(f.name)
        with open(f.name) as fin:
            self.assertEqual(self.CONLL, fin.read())
        os.remove(f.name)

    def test_convert_new(self):
        d = self.mock
        dd = d.convert({'Name': str.lower}, in_place=False)

        # made a copy
        self.assertIsNot(d, dd)

        # doesn't change original
        self.assertDictEqual(self.MOCK, d.fields)

        # changes copy
        self.assertDictEqual({'Name': ['alice', 'bob', 'carol'], 'SSN': ['123', None, '7890']}, dd.fields)

    def test_convert_in_place(self):
        d = self.mock
        dd = d.convert({'Name': str.lower}, in_place=True)

        # did not make a copy
        self.assertIs(d, dd)

        # changes original
        self.assertDictEqual({'Name': ['alice', 'bob', 'carol'], 'SSN': ['123', None, '7890']}, d.fields)

    def test_shuffle(self):
        d = self.mock
        dd = d.shuffle()
        self.assertIs(d, dd)

        # this relies on random seed
        self.assertDictEqual({'Name': ['Carol', 'Bob', 'Alice'], 'SSN': ['7890', None, '123']}, d.fields)

    def test_getitem(self):
        d = self.mock
        self.assertRaises(IndexError, lambda: d.__getitem__(10))
        self.assertDictEqual({'Name': 'Bob', 'SSN': None}, d[1])
        self.assertDictEqual({'Name': 'Alice', 'SSN': '123'}, d[0])
        self.assertDictEqual({'Name': 'Carol', 'SSN': '7890'}, d[-1])

        self.assertDictEqual({'Name': ['Bob', 'Carol'], 'SSN': [None, '7890']}, d[1:])
        self.assertDictEqual({'Name': ['Alice', 'Bob'], 'SSN': ['123', None]}, d[:2])
        self.assertDictEqual({'Name': ['Bob'], 'SSN': [None]}, d[1:2])

    def test_setitem(self):
        d = self.mock
        self.assertRaises(InvalidFieldsException, lambda: d.__setitem__(1, 'foo'))
        self.assertRaises(IndexError, lambda: d.__setitem__(10, {'Name': 'Victor', 'SSN': 123}))
        d[1] = {'Name': 'Victor', 'SSN': 123}
        self.assertDictEqual({'Name': ['Alice', 'Victor', 'Carol'], 'SSN': ['123', 123, '7890']}, d.fields)

    def test_copy(self):
        d = self.mock
        dd = d.copy()
        self.assertIsNot(d, dd)
        self.assertDictEqual(self.MOCK, d.fields)
        self.assertDictEqual(self.MOCK, dd.fields)

        for name in d.fields.keys():
            self.assertIsNot(d.fields[name], dd.fields[name])
