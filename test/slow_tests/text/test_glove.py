__author__ = 'victor'

import numpy as np
from unittest import TestCase
from stanza.text.vocab import GloveVocab


class TestGlove(TestCase):

    def test_get_embeddings(self):
        v = GloveVocab()
        v.add("!")
        e_exclamation = np.array([float(e) for e in """
        -0.58402 0.39031 0.65282 -0.3403 0.19493 -0.83489 0.11929 -0.57291 -0.56844 0.72989 -0.56975 0.53436 -0.38034 0.22471
        0.98031 -0.2966 0.126 0.55222 -0.62737 -0.082242 -0.085359 0.31515 0.96077 0.31986 0.87878 -1.5189 -1.7831 0.35639
        0.9674 -1.5497 2.335 0.8494 -1.2371 1.0623 -1.4267 -0.49056 0.85465 -1.2878 0.60204 -0.35963 0.28586 -0.052162
        -0.50818 -0.63459 0.33889 0.28416 -0.2034 -1.2338 0.46715 0.78858
        """.split() if e])
        E = v.get_embeddings(corpus='wikipedia_gigaword', n_dim=50)
        self.assertTrue(np.allclose(e_exclamation, E[v["!"]]))

