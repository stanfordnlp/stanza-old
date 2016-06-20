__author__ = 'kelvinguu'

from contextlib import contextmanager
import logging
import numpy as np
from stanza.text import Vocab


class Embeddings(object):
    """A map from strings to vectors.

    Vectors are efficiently stored in a matrix.
    """
    def __init__(self, array, vocab):
        """Create embeddings object.

        :param (np.array) array: has shape (vocab_size, embed_dim)
        :param (Vocab) vocab: a Vocab object
        """
        assert array.shape[0] == len(vocab)  # entries line up
        self.array = array
        self.vocab = vocab

    def __getitem__(self, w):
        idx = self.vocab.word2index(w)
        return self.array[idx]

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
        vocab = Vocab(unk)
        vocab.update(d)
        vecs = [None] * len(vocab)
        for key, vec in d.iteritems():
            vecs[vocab[key]] = vec
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
        print 'Starting'
        with open(array_path, mode) as array_file, open(vocab_path, mode) as vocab_file:
            yield array_file, vocab_file
        print 'Done'

    def to_file_path(self, path_prefix):
        """Write the embedding matrix and the vocab to <path_prefix>.npy and <path_prefix>.vocab.

        :param (str) path_prefix: path prefix of the saved files
        """
        with self._path_prefix_to_files(path_prefix, 'w') as (array_file, vocab_file):
            self.to_files(array_file, vocab_file)

    @classmethod
    def from_file_path(self, path_prefix):
        """Load the embedding matrix and the vocab from <path_prefix>.npy and <path_prefix>.vocab.

        :param (str) path_prefix: path prefix of the saved files
        """
        with self._path_prefix_to_files(path_prefix, 'r') as (array_file, vocab_file):
            return self.from_files(array_file, vocab_file)
