__author__ = 'victor'

import numpy as np
from unittest import TestCase
from stanza.text.vocab import SennaVocab


class TestSenna(TestCase):

    def test_get_embeddings(self):
        v = SennaVocab()
        v.add("!")
        E = v.get_embeddings()
        e_exclamation = np.array([float(e) for e in """
        -1.03682 1.77856 -0.693547 1.5948 1.5799 0.859243 1.15221 -0.976317 0.745304 -0.494589 0.308086 0.25239
        -0.1976 1.26203 0.813864 -0.940734 -0.215163 0.11645 0.525697 1.95766 0.394232 1.27717 0.710788 -0.389351
        0.161775 -0.106038 1.14148 0.607948 0.189781 -1.06022 0.280702 0.0251156 -0.198067 2.33027 0.408584
        0.350751 -0.351293 1.77318 -0.723457 -0.13806 -1.47247 0.541779 -2.57005 -0.227714 -0.817816 -0.552209
        0.360149 -0.10278 -0.36428 -0.64853
        """.split()])
        self.assertTrue(np.allclose(e_exclamation, E[v["!"]]))
