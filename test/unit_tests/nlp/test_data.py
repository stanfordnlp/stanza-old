import copy

import pytest

from stanza.nlp.corenlp import AnnotatedDocument, AnnotatedToken, AnnotatedSentence


@pytest.fixture
def json_dict():
    """What CoreNLP would return for 'Belgian swimmers beat the United States. Really?'"""
    return {u'sentences': [{u'index': 0,
                            u'parse': u'SENTENCE_SKIPPED_OR_UNPARSABLE',
                            u'tokens': [{u'after': u' ',
                                         u'before': u'',
                                         u'characterOffsetBegin': 0,
                                         u'characterOffsetEnd': 7,
                                         u'index': 1,
                                         u'originalText': u'Belgian',
                                         u'word': u'Belgian'},
                                        {u'after': u' ',
                                         u'before': u' ',
                                         u'characterOffsetBegin': 8,
                                         u'characterOffsetEnd': 16,
                                         u'index': 2,
                                         u'originalText': u'swimmers',
                                         u'word': u'swimmers'},
                                        {u'after': u' ',
                                         u'before': u' ',
                                         u'characterOffsetBegin': 17,
                                         u'characterOffsetEnd': 21,
                                         u'index': 3,
                                         u'originalText': u'beat',
                                         u'word': u'beat'},
                                        {u'after': u' ',
                                         u'before': u' ',
                                         u'characterOffsetBegin': 22,
                                         u'characterOffsetEnd': 25,
                                         u'index': 4,
                                         u'originalText': u'the',
                                         u'word': u'the'},
                                        {u'after': u' ',
                                         u'before': u' ',
                                         u'characterOffsetBegin': 26,
                                         u'characterOffsetEnd': 32,
                                         u'index': 5,
                                         u'originalText': u'United',
                                         u'word': u'United'},
                                        {u'after': u'',
                                         u'before': u' ',
                                         u'characterOffsetBegin': 33,
                                         u'characterOffsetEnd': 39,
                                         u'index': 6,
                                         u'originalText': u'States',
                                         u'word': u'States'},
                                        {u'after': u' ',
                                         u'before': u'',
                                         u'characterOffsetBegin': 39,
                                         u'characterOffsetEnd': 40,
                                         u'index': 7,
                                         u'originalText': u'.',
                                         u'word': u'.'}]},
                           {u'index': 1,
                            u'parse': u'SENTENCE_SKIPPED_OR_UNPARSABLE',
                            u'tokens': [{u'after': u'',
                                         u'before': u' ',
                                         u'characterOffsetBegin': 41,
                                         u'characterOffsetEnd': 47,
                                         u'index': 1,
                                         u'originalText': u'Really',
                                         u'word': u'Really'},
                                        {u'after': u'',
                                         u'before': u'',
                                         u'characterOffsetBegin': 47,
                                         u'characterOffsetEnd': 48,
                                         u'index': 2,
                                         u'originalText': u'?',
                                         u'word': u'?'}]}]}


class TestAnnotatedToken(object):
    def test_json_to_pb(self, json_dict):
        token_dict = json_dict['sentences'][0]['tokens'][0]
        token = AnnotatedToken.json_to_pb(token_dict)
        assert token.after == u' '
        assert token.before == u''
        assert token.beginChar == 0
        assert token.endChar == 7
        assert token.originalText == u'Belgian'
        assert token.word == u'Belgian'


class TestAnnotatedSentence(object):
    def test_json_to_pb(self, json_dict):
        orig_text = 'Really?'
        sent_dict = json_dict['sentences'][1]
        sent = AnnotatedSentence.json_to_pb(sent_dict)
        assert sent.text == orig_text
        assert sent.token[1].word == u'?'


class TestAnnotatedDocument(object):
    def test_json_to_pb(self, json_dict):
        orig_text = 'Belgian swimmers beat the United States. Really?'
        doc = AnnotatedDocument.json_to_pb(json_dict)
        assert doc.text == orig_text
        assert doc.sentence[1].text == 'Really?'

    def test_json(self, json_dict):
        doc = AnnotatedDocument.from_json(json_dict)
        new_json = doc.to_json()
        assert json_dict == new_json

    def test_eq(self, json_dict):
        # exact copy
        json_dict1 = copy.deepcopy(json_dict)

        # same as json_dict, but 'Belgian' is no longer capitalized
        json_dict2 = copy.deepcopy(json_dict)
        first_token_json = json_dict2['sentences'][0]['tokens'][0]
        first_token_json[u'originalText'] = 'belgian'
        first_token_json[u'word'] = 'belgian'

        doc = AnnotatedDocument.from_json(json_dict)
        doc1 = AnnotatedDocument.from_json(json_dict1)
        doc2 = AnnotatedDocument.from_json(json_dict2)

        assert doc == doc1
        assert doc != doc2

    @pytest.fixture
    def doc(self, json_dict):
        return AnnotatedDocument.from_json(json_dict)

    def test_properties(self, doc):
        assert doc[0][1].word == u'swimmers'
        assert doc[0][2].character_span == (17, 21)
        assert doc[0].document == doc

