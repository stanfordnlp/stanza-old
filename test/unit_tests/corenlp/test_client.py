import pytest

from stanza.corenlp.client import AnnotatedDocument


@pytest.fixture
def json_dict():
  """What CoreNLP would return for 'Belgian swimmers beat the United States.'"""
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
                                      {u'after': u'',
                                       u'before': u'',
                                       u'characterOffsetBegin': 39,
                                       u'characterOffsetEnd': 40,
                                       u'index': 7,
                                       u'originalText': u'.',
                                       u'word': u'.'}]}]}


def test_dict_to_pb(json_dict):
  doc = AnnotatedDocument.dict_to_pb(json_dict)
  # TODO(kelvin): flesh out this test
  assert False