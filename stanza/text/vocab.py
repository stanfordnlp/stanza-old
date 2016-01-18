__author__ = 'victor'
from collections import Counter, namedtuple, OrderedDict
from itertools import izip
import numpy as np
import zipfile
from ..util.resource import get_data_or_download


class Vocab(object):
    """Defines a bijection between N words and the integers 0 through N-1."""

    def __init__(self, unk):
        """Construct a Vocab object.

        Args:
            unk: a string to represent the unknown word (UNK).
        """
        self.__word2index = OrderedDict()
        self.__counts = Counter()
        self.__unk = unk

        # assign an index for UNK
        self.add(self.__unk, count=0)

    def __iter__(self):
        return iter(self.__word2index)

    def __repr__(self):
        """Represent Vocab as a dictionary from words to indices."""
        return str(self.__word2index)

    def __str__(self):
        return 'Vocab(%d words)' % len(self.__word2index)

    def __len__(self):
        """Get total number of entries in vocab."""
        return len(self.__word2index)

    def __getitem__(self, word):
        """Get the index for a word.

        If the word is unknown, the index for UNK is returned.
        """
        try:
            return self.__word2index[word]
        except KeyError:
            return self.__word2index[self.__unk]

    def __contains__(self, word):
        return word in self.__word2index

    def add(self, word, count=1):
        """Add a word to the vocabulary and return its index.

        Also, updates the counts for a word.

        WARNING: this function assumes that if the Vocab currently has N words, then
        there is a perfect bijection between these N words and the integers 0 through N-1.
        """
        if word not in self.__word2index:
            self.__word2index[word] = len(self.__word2index)
        self.__counts[word] += count
        return self.__word2index[word]

    def update(self, words):
        """Add an iterable of words to the Vocabulary."""
        return [self.add(w) for w in words]

    def words2indices(self, words):
        """Convert a list of words into a list of indices."""
        return [self[w] for w in words]

    def indices2words(self, indices):
        """Converts a list of indices into a list of words."""
        index2word = self.__word2index.keys()  # works because word2index is an OrderedDict
        return [index2word[i] for i in indices]

    def prune_rares(self, cutoff=2):
        """
        returns a **new** `Vocab` object that is similar to this one but
        with words occurring less than `cutoff` times removed. Note that
        the indices in the new `Vocab` will be remapped (because rare
        words will have been removed).

        NOTE: UNK is never pruned.
        """
        v = self.__class__(unk=self.__unk)  # use __class__ to support subclasses
        for w in self.__word2index:
            if self.__counts[w] >= cutoff or w == self.__unk:  # don't remove unk
                v.add(w, count=self.__counts[w])
        return v

    def sort_by_decreasing_count(self):
        """Return a **new** `Vocab` object that is ordered by decreasing count.

        The word at index 1 will be most common, the word at index 2 will be
        next most common, and so on.

        NOTE: UNK will remain at index 0, regardless of its frequency.
        """
        v = self.__class__(unk=self.__unk)  # use __class__ to support subclasses

        # UNK gets index 0
        v.add(self.__unk, count=self.__counts[self.__unk])

        for word, count in self.__counts.most_common():
            if word != self.__unk:
                v.add(word, count=count)
        return v

    def clear_counts(self):
        """Removes counts for all tokens."""
        self.__counts.clear()
        # TODO: this removes the entries too, rather than setting them to 0

    @classmethod
    def from_dict(cls, word2index, unk):
        """Create Vocab from an existing string to integer dictionary.

        All counts are set to 0.

        Args:
            word2index: a dictionary representing a bijection from N words to the integers 0 through N-1.
                UNK must be assigned the 0 index.
            unk: the string representing unk in word2index.
        """
        try:
            if word2index[unk] != 0:
                raise ValueError('unk must be assigned index 0')
        except KeyError:
            raise ValueError('word2index must have an entry for unk.')

        # check that word2index is a bijection
        vals = set(word2index.values())  # unique indices
        n = len(vals)

        bijection = (len(word2index) == n) and (vals == set(range(n)))
        if not bijection:
            raise ValueError('word2index is not a bijection between N words and the integers 0 through N-1.')

        # reverse the dictionary
        index2word = {idx: word for word, idx in word2index.iteritems()}

        vocab = cls(unk=unk)
        for i in xrange(n):
            vocab.add(index2word[i])

        return vocab


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
