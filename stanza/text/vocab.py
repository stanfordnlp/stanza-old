__author__ = 'victor'
from collections import Counter
import numpy as np
from itertools import izip
from ..util.resource import get_data_or_download


class Vocab(object):
    """
    An abstraction for a vocabulary object that maps between words and integer indices.
    """

    def __init__(self, unk=''):
        self.word2index = {}
        self.index2word = []
        self.counts = Counter()
        self.unk = unk

        if self.unk:
            self.add(self.unk, 0)

    def clear_counts(self):
        """
        removes counts for all tokens
        """
        self.counts.clear()

    def __repr__(self):
        return str(self.word2index)

    def __len__(self):
        return len(self.index2word)

    def __getitem__(self, word):
        if self.unk:
            return self.word2index.get(word, self.word2index[self.unk])
        else:
            return self.word2index[word]

    def __contains__(self, word):
        return word in self.word2index

    def add(self, word, count=1):
        """
        add a word to the vocabulary and return its index
        """
        if word not in self.word2index:
            self.word2index[word] = len(self)
            self.index2word.append(word)
        self.counts[word] += count
        return self.word2index[word]

    def words2indices(self, words, add=False):
        """
        converts a list of words into a list of indices. If `add` is `True`
        then unknown words will be added to the vocabulary
        """
        if add:
            return [self.add(w) for w in words]
        else:
            return [self[w] for w in words]

    def indices2words(self, indices):
        """
        converts a list of indices into a list of words.
        """
        return [self.index2word[i] for i in indices]

    def prune_rares(self, cutoff=2):
        """
        returns a **new** `Vocab` object that is similar to this one but
        with words occurring less than `cutoff` times removed. Note that
        the indices in the new `Vocab` will be remapped (because rare
        words will have been removed).
        """
        v = self.__class__(unk=self.unk) # use __class__ to support subclasses
        for w in self.index2word:
            if self.counts[w] >= cutoff or w == self.unk: # don't remove unk
                v.add(w, count=self.counts[w])
        return v

    def sort_by_decreasing_count(self):
        """
        returns a **new** `Vocab` object that is ordered by decreasing count.
        That is, the word at index 0 is the most common and so forth. If unknown
        is supported, then the most common word is at index 1 and `unk` remains
        in index 0.
        """
        v = self.__class__(unk=self.unk) # use __class__ to support subclasses
        if self.unk:
            v.add(self.unk, count=self.counts[self.unk])
        for word, count in self.counts.most_common():
            if word != self.unk:
                v.add(word, count=count)
        return v


class EmbeddedVocab(Vocab):

    def __init__(self, unk):
        super(EmbeddedVocab, self).__init__(unk=unk)

    def get_embeddings(self):
        raise NotImplementedError()

    @classmethod
    def gen_word_list(cls, fname):
        with open(fname) as f:
            for line in f:
                yield line.rstrip("\n\r")

    @classmethod
    def gen_embeddings(cls, fname):
        with open(fname) as f:
            for line in f:
                yield np.fromstring(line, sep=' ')


class SennaVocab(EmbeddedVocab):

    embeddings_url = 'https://github.com/baojie/senna/raw/master/embeddings/embeddings.txt'
    words_url = 'https://raw.githubusercontent.com/baojie/senna/master/hash/words.lst'
    n_dim = 50

    def __init__(self, unk='UNKNOWN'):
        super(SennaVocab, self).__init__(unk=unk)

    def get_embeddings(self, rand=None):
        rand = rand if rand else lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        embeddings = get_data_or_download('senna', 'embeddings.txt', self.embeddings_url)
        words = get_data_or_download('senna', 'words.lst', self.words_url)

        E = rand((len(self), self.n_dim))

        for word_emb in izip(self.gen_word_list(words), self.gen_embeddings(embeddings)):
            w, e = word_emb
            if w in self:
                E[self.word2index[w]] = e
        return E
