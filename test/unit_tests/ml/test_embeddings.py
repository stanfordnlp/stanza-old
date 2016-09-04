import pytest

from stanza.ml.embeddings import Embeddings
from stanza.text import Vocab
import numpy as np
from numpy.testing import assert_approx_equal


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

    knn = embeddings.k_nearest(query, 3)
    assert knn == [('show', 58), ('a', 40), ('what', 22)]


def test_k_nearest_approx(embeddings):
    # Code for calculating the correct cosine similarities.
    # for i in range(len(array)):
    #    print 1-scipy.spatial.distance.cosine(array[i,:], query)

    query = np.array([3, 2, 1])
    knn = embeddings.k_nearest_approx(query, 3)
    correct = [('show', 0.89199106528525429), ('a', 0.87579576196887721), ('what', 0.83152184062029977)]
    assert len(knn) == len(correct)
    for (w1, s1), (w2, s2) in zip(knn, correct):
        assert w1 == w2
        assert_approx_equal(s1, s2)


def test_subset(embeddings):
    sub = embeddings.subset(['a', 'what'])
    assert sub.to_dict() == {'a': [6, 7, 8], 'unk': [0, 1, 2], 'what': [3, 4, 5]}
