__author__ = 'kelvinguu'

from contextlib import contextmanager
import logging
import numpy as np
from stanza.text import Vocab


class Embeddings(object):

    def __init__(self, array, vocab):
        """Create embeddings object.

        :param array: a numpy array
        :param vocab: a Vocab object
        """
        assert isinstance(array, np.ndarray)
        assert isinstance(vocab, Vocab)
        assert array.shape[0] == len(vocab)  # entries line up
        self.array = array
        self.vocab = vocab

    def to_dict(self):
        d = {}
        for word, idx in self.vocab.iteritems():
            d[word] = self.array[idx]
        return d

    def to_files(self, array_file, vocab_file):
        logging.info('Writing array...')
        np.save(array_file, self.array)
        logging.info('Writing vocab...')
        self.vocab.to_file(vocab_file)

    @classmethod
    def from_files(cls, array_file, vocab_file):
        logging.info('Loading array...')
        array = np.load(array_file)
        logging.info('Loading vocab...')
        vocab = Vocab.from_file(vocab_file)
        return cls(array, vocab)

    @staticmethod
    @contextmanager
    def path_prefix_to_files(path_prefix, mode):
        array_path = path_prefix + '.npy'
        vocab_path = path_prefix + '.vocab'
        print 'Starting'
        with open(array_path, mode) as array_file, open(vocab_path, mode) as vocab_file:
            yield array_file, vocab_file
        print 'Done'

    def to_file_path(self, path_prefix):
        with self.path_prefix_to_files(path_prefix, 'w') as (array_file, vocab_file):
            self.to_files(array_file, vocab_file)

    @classmethod
    def from_file_path(self, path_prefix):
        with self.path_prefix_to_files(path_prefix, 'r') as (array_file, vocab_file):
            return self.from_files(array_file, vocab_file)
