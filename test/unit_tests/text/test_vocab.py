__author__ = 'victor, kelvinguu'

from collections import Counter
from unittest import TestCase
from stanza.text.vocab import Vocab, SennaVocab, GloveVocab


class TestVocab(TestCase):

    def setUp(self):
        self.Vocab = Vocab

    def test_unk(self):
        unk = '**UNK**'
        v = self.Vocab(unk=unk)
        self.assertEqual(len(v), 1)
        self.assertIn(unk, v)
        self.assertEqual(v[unk], 0)
        self.assertEqual(v.count(unk), 0)

    def test_add(self):
        v = self.Vocab('**UNK**')
        v.add('hi')
        self.assertIn('hi', v)
        self.assertEqual(len(v), 2)
        self.assertEqual(v.count('hi'), 1)
        self.assertEqual(v['hi'], 1)

    def test_sent2index(self):
        v = self.Vocab(unk='unk')
        words = ['i', 'like', 'pie']
        v.update(words)
        self.assertEqual(v.words2indices(words), [1, 2, 3])
        self.assertEqual(v.words2indices(['i', 'said']), [1, 0])

    def test_index2sent(self):
        v = self.Vocab(unk='unk')
        v.update(['i', 'like', 'pie'])
        words = v.indices2words([1, 2, 3, 0])
        self.assertEqual(words, ['i', 'like', 'pie', 'unk'])

    def test_prune_rares(self):
        v = self.Vocab(unk='unk')
        v.update(['hi'] * 3 + ['bye'] * 5)
        self.assertEqual({'hi': 3, 'bye': 5, 'unk': 0}, dict(v.counts_copy()))
        p = v.prune_rares(cutoff=4)
        self.assertEqual({'bye': 5, 'unk': 0}, dict(p.counts_copy()))

    def test_sort_by_decreasing_count(self):
        v = self.Vocab(unk='unk')
        v.update('some words words for for for you you you you'.split())
        s = v.sort_by_decreasing_count()
        self.assertEqual(['unk', 'you', 'for', 'words', 'some'], list(iter(s)))
        self.assertEqual({'unk': 0, 'you': 4, 'for': 3, 'words': 2, 'some': 1}, dict(s.counts_copy()))

    def test_from_file(self):
        lines = ['unk\t10\n', 'cat\t4\n', 'bear\t6']
        vocab = Vocab.from_file(lines)
        self.assertEqual(vocab.counts_copy(), Counter({'unk': 10, 'cat': 4, 'bear': 6}))
        self.assertEqual(dict(vocab._word2index), {'unk': 0, 'cat': 1, 'bear': 2})


class TestSenna(TestVocab):

    def setUp(self):
        self.Vocab = SennaVocab


class TestGlove(TestVocab):

    def setUp(self):
        self.Vocab = GloveVocab
