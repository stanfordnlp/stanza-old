import pytest

from stanza.ml.embeddings import Embeddings
from stanza.text import Vocab
import numpy as np


@pytest.fixture
def embeddings():
    v = Vocab('unk')
    v.update('what a show'.split())
    array = np.reshape(np.arange(12), (4, 3))
    return Embeddings(array, v)


def test_to_dict(embeddings):
    correct = {'a': [6, 7, 8],
               'show': [9, 10, 11],
               'unk': [0, 1, 2],
               'what': [3, 4, 5]}
    d = embeddings.to_dict()
    d = {k: v.tolist() for k, v in d.items()}  # convert arrays to list for easy comparison
    assert d == correct