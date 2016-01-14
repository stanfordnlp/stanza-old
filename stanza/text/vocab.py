__author__ = 'victor'
from collections import Counter, namedtuple
from itertools import izip
import numpy as np
import zipfile
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
        v = self.__class__(unk=self.unk)  # use __class__ to support subclasses
        for w in self.index2word:
            if self.counts[w] >= cutoff or w == self.unk:  # don't remove unk
                v.add(w, count=self.counts[w])
        return v

    def sort_by_decreasing_count(self):
        """
        returns a **new** `Vocab` object that is ordered by decreasing count.
        That is, the word at index 0 is the most common and so forth. If unknown
        is supported, then the most common word is at index 1 and `unk` remains
        in index 0.
        """
        v = self.__class__(unk=self.unk)  # use __class__ to support subclasses
        if self.unk:
            v.add(self.unk, count=self.counts[self.unk])
        for word, count in self.counts.most_common():
            if word != self.unk:
                v.add(word, count=count)
        return v


class EmbeddedVocab(Vocab):

    def get_embeddings(self):
        raise NotImplementedError()

    def backfill_unk_emb(self, E, filled_words):
        if self.unk:
            unk_emb = E[self.word2index[self.unk]]
            for i, word in enumerate(self.index2word):
                if word not in filled_words:
                    E[i] = unk_emb


class SennaVocab(EmbeddedVocab):

    embeddings_url = 'https://github.com/baojie/senna/raw/master/embeddings/embeddings.txt'
    words_url = 'https://raw.githubusercontent.com/baojie/senna/master/hash/words.lst'
    n_dim = 50

    def __init__(self, unk='UNKNOWN'):
        super(SennaVocab, self).__init__(unk=unk)

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

    def get_embeddings(self, rand=None, dtype='float32'):
        rand = rand if rand else lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        embeddings = get_data_or_download('senna', 'embeddings.txt', self.embeddings_url)
        words = get_data_or_download('senna', 'words.lst', self.words_url)

        E = rand((len(self), self.n_dim)).astype(dtype)

        seen = []
        for word_emb in izip(self.gen_word_list(words), self.gen_embeddings(embeddings)):
            w, e = word_emb
            if w in self:
                seen += [w]
                E[self.word2index[w]] = e
        self.backfill_unk_emb(E, set(seen))
        return E


class GloveVocab(EmbeddedVocab):

    GloveSetting = namedtuple('GloveSetting', ['url', 'n_dims', 'size', 'description'])
    settings = {
        'common_crawl_48': GloveSetting('http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                        [300], '1.75GB', '48B token common crawl'),
        'common_crawl_840': GloveSetting('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                         [300], '2.03GB', '840B token common crawl'),
        'twitter': GloveSetting('http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                                [25, 50, 100, 200], '1.42GB', '27B token twitter'),
        'wikipedia_gigaword': GloveSetting('http://nlp.stanford.edu/data/glove.6B.zip',
                                           [50, 100, 200, 300], '822MB', '6B token wikipedia 2014 + gigaword 5'),
    }

    def __init__(self, unk='', corpus='common_crawl_48', n_dim=300):
        super(GloveVocab, self).__init__(unk)
        assert corpus in self.settings, '{} not in supported corpus {}'.format(corpus, self.settings.keys())
        self.n_dim, self.corpus, self.setting = n_dim, corpus, self.settings[corpus]
        assert n_dim in self.setting.n_dims, '{} not in supported dimensions {}'.format(n_dim, self.setting.n_dims)

    def get_embeddings(self, rand=None, dtype='float32'):
        rand = rand if rand else lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        zip_file = get_data_or_download('glove', '{}.zip'.format(self.corpus), self.setting.url, size=self.setting.size)

        E = rand((len(self), self.n_dim)).astype(dtype)
        n_dim = str(self.n_dim)

        with zipfile.ZipFile(open(zip_file)) as zf:
            # should be only 1 txt file
            names = [info.filename for info in zf.infolist() if info.filename.endswith('.txt') and n_dim in info.filename]
            if not names:
                s = 'no .txt files found in zip file that matches {}-dim!'.format(n_dim)
                s += '\n available files: {}'.format(names)
                raise IOError(s)
            name = names[0]
            seen = []
            with zf.open(name) as f:
                for line in f:
                    toks = line.rstrip().split(' ')
                    word = toks[0]
                    if word in self:
                        seen += [word]
                        E[self.word2index[word]] = np.array([float(w) for w in toks[1:]], dtype=dtype)
            self.backfill_unk_emb(E, set(seen))
            return E
