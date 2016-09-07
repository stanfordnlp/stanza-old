from collections import Mapping
from contextlib import contextmanager
import logging
import numpy as np
from stanza.text import Vocab


__author__ = 'kelvinguu'


class Embeddings(Mapping):
    """A map from strings to vectors.

    Vectors are stored as a numpy array.
    Vectors are saved/loaded from disk using numpy.load, which is roughly 3-4 times faster
    than reading a text file.
    """

    def __init__(self, array, vocab):
        """Create embeddings object.

        :param (np.array) array: has shape (vocab_size, embed_dim)
        :param (Vocab) vocab: a Vocab object
        """
        assert len(array.shape) == 2
        assert array.shape[0] == len(vocab)  # entries line up

        self.array = array
        self.vocab = vocab

    def __getitem__(self, w):
        idx = self.vocab.word2index(w)
        return self.array[idx]

    def __iter__(self):
        return iter(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, item):
        return item in self.vocab

    def subset(self, words):
        sub_vocab = self.vocab.subset(words)
        idxs = [self.vocab[w] for w in sub_vocab]
        sub_array = self.array[idxs]
        return self.__class__(sub_array, sub_vocab)

    def inner_products(self, vec):
        """Get the inner product of a vector with every embedding.

        :param (np.array) vector: the query vector

        :return (list[tuple[str, float]]): a map of embeddings to inner products
        """
        products = self.array.dot(vec)
        return self._word_to_score(np.arange(len(products)), products)

    def _word_to_score(self, ids, scores):
        """Return a map from each word to its score.

        :param (np.array) ids: a vector of word ids
        :param (np.array) scores: a vector of scores

        :return (dict[unicode, float]): a map from each word (unicode) to its score (float)
        """
        # should be 1-D vectors
        assert len(ids.shape) == 1
        assert ids.shape == scores.shape

        w2s = {}
        for i in range(len(ids)):
            w2s[self.vocab.index2word(ids[i])] = scores[i]
        return w2s

    def k_nearest(self, vec, k):
        """Get the k nearest neighbors of a vector (in terms of highest inner products).

        :param (np.array) vec: query vector
        :param (int) k: number of top neighbors to return

        :return (list[tuple[str, float]]): a list of (word, score) pairs, in descending order
        """
        nbr_score_pairs = self.inner_products(vec)
        return sorted(nbr_score_pairs.items(), key=lambda x: x[1], reverse=True)[:k]

    def _init_lsh_forest(self):
        """Construct an LSH forest for nearest neighbor search."""
        import sklearn.neighbors
        lshf = sklearn.neighbors.LSHForest()
        lshf.fit(self.array)
        return lshf

    def k_nearest_approx(self, vec, k):
        """Get the k nearest neighbors of a vector (in terms of cosine similarity).

        :param (np.array) vec: query vector
        :param (int) k: number of top neighbors to return

        :return (list[tuple[str, float]]): a list of (word, cosine similarity) pairs, in descending order
        """
        if not hasattr(self, 'lshf'):
            self.lshf = self._init_lsh_forest()

        # TODO(kelvin): make this inner product score, to be consistent with k_nearest
        distances, neighbors = self.lshf.kneighbors(vec, n_neighbors=k, return_distance=True)
        scores = np.subtract(1, distances)
        nbr_score_pairs = self._word_to_score(np.squeeze(neighbors), np.squeeze(scores))

        return sorted(nbr_score_pairs.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self):
        """Convert to dictionary.

        :return (dict): A dict mapping from strings to vectors.
        """
        d = {}
        for word, idx in self.vocab.iteritems():
            d[word] = self.array[idx].tolist()
        return d

    @classmethod
    def from_dict(cls, d, unk):
        assert unk in d
        vocab = Vocab(unk)
        vocab.update(d)
        vecs = []
        for i in range(len(vocab)):
            word = vocab.index2word(i)
            vec = d[word]
            vecs.append(vec)
        array = np.array(vecs)
        return cls(array, vocab)

    def to_files(self, array_file, vocab_file):
        """Write the embedding matrix and the vocab to files.

        :param (file) array_file: file to write array to
        :param (file) vocab_file: file to write vocab to
        """
        logging.info('Writing array...')
        np.save(array_file, self.array)
        logging.info('Writing vocab...')
        self.vocab.to_file(vocab_file)

    @classmethod
    def from_files(cls, array_file, vocab_file):
        """Load the embedding matrix and the vocab from files.

        :param (file) array_file: file to read array from
        :param (file) vocab_file: file to read vocab from

        :return (Embeddings): an Embeddings object
        """
        logging.info('Loading array...')
        array = np.load(array_file)
        logging.info('Loading vocab...')
        vocab = Vocab.from_file(vocab_file)
        return cls(array, vocab)

    @staticmethod
    @contextmanager
    def _path_prefix_to_files(path_prefix, mode):
        array_path = path_prefix + '.npy'
        vocab_path = path_prefix + '.vocab'
        with open(array_path, mode) as array_file, open(vocab_path, mode) as vocab_file:
            yield array_file, vocab_file

    def to_file_path(self, path_prefix):
        """Write the embedding matrix and the vocab to <path_prefix>.npy and <path_prefix>.vocab.

        :param (str) path_prefix: path prefix of the saved files
        """
        with self._path_prefix_to_files(path_prefix, 'w') as (array_file, vocab_file):
            self.to_files(array_file, vocab_file)

    @classmethod
    def from_file_path(cls, path_prefix):
        """Load the embedding matrix and the vocab from <path_prefix>.npy and <path_prefix>.vocab.

        :param (str) path_prefix: path prefix of the saved files
        """
        with cls._path_prefix_to_files(path_prefix, 'r') as (array_file, vocab_file):
            return cls.from_files(array_file, vocab_file)
