__author__ = 'victor'

from unittest import TestCase
from stanza.text.vocab import Vocab


class TestVocab(TestCase):

    def test_unk(self):
        v = Vocab() # no unk
        self.assertEqual(len(v), 0)

        v = Vocab(unk='**UNK**')
        self.assertEqual(len(v), 1)
        self.assertIn(v.unk, v)
        self.assertEqual(v[v.unk], 0)
        self.assertEqual(v.counts[v.unk], 0)
        self.assertEqual(v.unk, '**UNK**')

    def test_add(self):
        v = Vocab()
        v.add('hi')
        self.assertIn('hi', v)
        self.assertEqual(len(v), 1)
        self.assertEqual(v.counts['hi'], 1)
        self.assertEqual(v['hi'], 0)

    def test_sent2index(self):
        v = Vocab(unk='unk')
        inds = v.words2indices(['i', 'like', 'pie'], add=True)
        self.assertEqual(inds, [1, 2, 3])
        inds = v.words2indices(['i', 'said'], add=True)
        self.assertEqual(inds, [1, 4])
        inds = v.words2indices(['you', 'said'], add=False)
        self.assertEqual(inds, [0, 4])

    def test_index2sent(self):
        v = Vocab(unk='unk')
        inds = v.words2indices(['i', 'like', 'pie'], add=True)
        words = v.indices2words([1, 2, 3, 0])
        self.assertEqual(words, ['i', 'like', 'pie', v.unk])

    def test_prune_rares(self):
        v = Vocab(unk='unk')
        v.words2indices(['hi'] * 3 + ['bye'] * 5, add=True)
        self.assertEqual({'hi': 3, 'bye': 5, v.unk: 0}, v.counts)
        p = v.prune_rares(cutoff=4)
        self.assertEqual({'bye': 5, v.unk: 0}, p.counts)

    def test_sort_by_decreasing_count(self):
        v = Vocab(unk='unk')
        v.words2indices('some words words for for for you you you you'.split(), add=True)
        s = v.sort_by_decreasing_count()
        self.assertEqual(['unk', 'you', 'for', 'words', 'some'], s.index2word)
        self.assertEqual({'unk': 0, 'you': 4, 'for': 3, 'words': 2, 'some': 1}, s.counts)

        # same test, no unk
        v = Vocab()
        v.words2indices('some words words for for for you you you you'.split(), add=True)
        s = v.sort_by_decreasing_count()
        self.assertEqual(['you', 'for', 'words', 'some'], s.index2word)
        self.assertEqual({'you': 4, 'for': 3, 'words': 2, 'some': 1}, s.counts)
