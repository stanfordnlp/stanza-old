#
# pylint: disable=no-self-use, redefined-outer-name

import copy
import json

import pytest

import stanza.nlp.CoreNLP_pb2 as proto

from stanza.nlp.corenlp import AnnotatedDocument, AnnotatedToken, AnnotatedSentence


@pytest.fixture
def json_dict():
    """What CoreNLP would return for 'Belgian swimmers beat the United States. Really?'"""
    return json.loads('{"text": "Belgian swimmers beat the United States. Really?", "sentence": [{"characterOffsetBegin": 0, "hasRelationAnnotations": false, "hasNumerizedTokensAnnotation": false, "tokenOffsetBegin": 0, "token": [{"before": "", "value": "Belgian", "hasXmlContext": false, "endChar": 7, "beginChar": 0, "after": " ", "originalText": "Belgian", "word": "Belgian"}, {"before": " ", "value": "swimmers", "hasXmlContext": false, "endChar": 16, "beginChar": 8, "after": " ", "originalText": "swimmers", "word": "swimmers"}, {"before": " ", "value": "beat", "hasXmlContext": false, "endChar": 21, "beginChar": 17, "after": " ", "originalText": "beat", "word": "beat"}, {"before": " ", "value": "the", "hasXmlContext": false, "endChar": 25, "beginChar": 22, "after": " ", "originalText": "the", "word": "the"}, {"before": " ", "value": "United", "hasXmlContext": false, "endChar": 32, "beginChar": 26, "after": " ", "originalText": "United", "word": "United"}, {"before": " ", "value": "States", "hasXmlContext": false, "endChar": 39, "beginChar": 33, "after": "", "originalText": "States", "word": "States"}, {"before": "", "value": ".", "hasXmlContext": false, "endChar": 40, "beginChar": 39, "after": " ", "originalText": ".", "word": "."}], "tokenOffsetEnd": 7, "sentenceIndex": 0, "characterOffsetEnd": 40}, {"characterOffsetBegin": 41, "hasRelationAnnotations": false, "hasNumerizedTokensAnnotation": false, "tokenOffsetBegin": 7, "token": [{"before": " ", "value": "Really", "hasXmlContext": false, "endChar": 47, "beginChar": 41, "after": "", "originalText": "Really", "word": "Really"}, {"before": "", "value": "?", "hasXmlContext": false, "endChar": 48, "beginChar": 47, "after": "", "originalText": "?", "word": "?"}], "tokenOffsetEnd": 9, "sentenceIndex": 1, "characterOffsetEnd": 48}]}')

@pytest.fixture
def document_pb():
    """What CoreNLP would return for:
       "Barack Hussein Obama is an American politician who is the 44th
       and current President of the United States. He is the first
       African American to hold the office and the first president born
       outside the continental United States. Born in Honolulu, Hawaii,
       Obama is a graduate of Columbia University and Harvard Law
       School, where he was president of the Harvard Law Review."
    """
    doc = proto.Document()
    with open("test/unit_tests/nlp/document.pb", "rb") as f:
        doc.ParseFromString(f.read())
    return doc

class TestAnnotatedToken(object):
    #def test_json_to_pb(self, json_dict):
    #    token_dict = json_dict['sentences'][0]['tokens'][0]
    #    token = AnnotatedToken.json_to_pb(token_dict)
    #    assert token.after == u' '
    #    assert token.before == u''
    #    assert token.beginChar == 0
    #    assert token.endChar == 7
    #    assert token.originalText == u'Belgian'
    #    assert token.word == u'Belgian'

    def test_parse_pb(self, document_pb):
        token_pb = document_pb.sentence[1].token[3]
        token = AnnotatedToken.from_pb(token_pb)
        assert token.after == u' '
        assert token.before == u' '
        assert token.character_span == (117, 122)
        assert token.originalText == u'first'
        assert token.word == u'first'
        assert token.lemma == u'first'
        assert token.ner == u'ORDINAL'
        assert token.pos == u'JJ'

class TestAnnotatedSentence(object):
    #def test_json_to_pb(self, json_dict):
    #    orig_text = 'Really?'
    #    sent_dict = json_dict['sentences'][1]
    #    sent = AnnotatedSentence.from_json(sent_dict)
    #    assert sent.text == orig_text
    #    assert sent[1].word == u'?'

    def test_parse_pb(self, document_pb):
        sentence_pb = document_pb.sentence[0]
        sentence = AnnotatedSentence.from_pb(sentence_pb)
        assert sentence.text == u"Barack Hussein Obama is an American politician who is the 44th and current President of the United States."
        assert len(sentence) == 19
        assert sentence[1].word == "Hussein"
        assert sentence[1].ner == "PERSON"

    def test_depparse(self, document_pb):
        sentence_pb = document_pb.sentence[0]
        sentence = AnnotatedSentence.from_pb(sentence_pb)
        dp = sentence.depparse()
        assert dp.roots == [6] # politician
        assert (2, 'nsubj') in dp.children(6) # Obama is child of politician
        assert (3, 'cop') in dp.children(6) # 'is' is ia copula
        assert (0, 'compound') in dp.children(2) # 'Barack' is part of the compount that is Obama.

    def test_depparse_json(self, document_pb):
        sentence_pb = document_pb.sentence[0]
        sentence = AnnotatedSentence.from_pb(sentence_pb)
        dp = sentence.depparse()
        edges = dp.to_json()
        # politician is root
        assert any((edge['dep'] == 'root' and edge['dependent'] == 7 and edge['dependentgloss'] == 'politician') for edge in edges)
        # Obama is child of politician
        assert any((edge['governer'] == 7 and edge['dep'] == 'nsubj' and edge['dependent'] == 3 and edge['dependentgloss'] == 'Obama') for edge in edges)
        # 'is' is ia copula
        assert any((edge['governer'] == 7 and edge['dep'] == 'cop' and edge['dependent'] == 4 and edge['dependentgloss'] == 'is') for edge in edges)
        # 'Barack' is part of the compount that is Obama.
        assert any((edge['governer'] == 3 and edge['dep'] == 'compound' and edge['dependent'] == 1 and edge['dependentgloss'] == 'Barack') for edge in edges)

    def test_from_tokens(self):
        text = "This is a test."
        tokens = "This is a test .".split()
        sentence = AnnotatedSentence.from_tokens(text, tokens)
        assert sentence.text == text
        assert len(sentence) == 5
        assert sentence[1].word == "is"

class TestAnnotatedDocument(object):
    #def test_json_to_pb(self, json_dict):
    #    orig_text = 'Belgian swimmers beat the United States. Really?'
    #    doc = AnnotatedDocument.from_json(json_dict)
    #    assert doc.text == orig_text
    #    assert doc[1].text == 'Really?'

    def test_json(self, json_dict):
        doc = AnnotatedDocument.from_json(json_dict)
        new_json = doc.to_json()
        assert json_dict == new_json

    def test_eq(self, json_dict):
        # exact copy
        json_dict1 = copy.deepcopy(json_dict)

        # same as json_dict, but 'Belgian' is no longer capitalized
        json_dict2 = copy.deepcopy(json_dict)
        first_token_json = json_dict2['sentence'][0]['token'][0]
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

    def test_parse_pb(self, document_pb):
        document = AnnotatedDocument.from_pb(document_pb)
        assert document.text == u"Barack Hussein Obama is an American politician who is the 44th and current President of the United States. He is the first African American to hold the office and the first president born outside the continental United States. Born in Honolulu, Hawaii, Obama is a graduate of Columbia University and Harvard Law School, where he was president of the Harvard Law Review."
        assert len(document) == 3
        assert document[0][1].word == "Hussein"
        assert document[0][1].ner == "PERSON"

    def test_mentions(self, document_pb):
        document = AnnotatedDocument.from_pb(document_pb)
        mentions = document.mentions
        assert len(mentions) == 17
