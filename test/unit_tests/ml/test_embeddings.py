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


@pytest.fixture
def dict_embeddings():
    return {'a': [6, 7, 8],
            'show': [9, 10, 11],
            'unk': [0, 1, 2],
            'what': [3, 4, 5]}


def test_to_dict(embeddings, dict_embeddings):
    d = embeddings.to_dict()
    assert d == dict_embeddings


def test_from_dict(embeddings, dict_embeddings):
    emb = Embeddings.from_dict(dict_embeddings, 'unk')
    assert emb.to_dict() == dict_embeddings


def test_get_item(embeddings):
    assert embeddings['what'].tolist() == [3, 4, 5]


def test_inner_products(embeddings):
    query = np.array([3, 2, 1])
    scores = embeddings.inner_products(query)
    correct = {
        'a': 18 + 14 + 8,
        'show': 27 + 20 + 11,
        'unk': 2 + 2,
        'what': 9 + 8 + 5,
    }
    assert scores == correct

    knn = embeddings.k_nearest_neighbors(query, 3)
    assert knn == [('show', 58), ('a', 40), ('what', 22)]


def test_subset(embeddings):
    sub = embeddings.subset(['a', 'what'])
    assert sub.to_dict() == {'a': [6, 7, 8], 'unk': [0, 1, 2], 'what': [3, 4, 5]}
