from abc import abstractproperty
from collections import Sequence

import requests
import CoreNLP_pb2
from google.protobuf.internal.decoder import _DecodeVarint

__author__ = 'kelvinguu, vzhong'


class Client(object):
  """
  A CoreNLP client to the Stanford CoreNLP server.
  """

  def __init__(self, server='http://localhost:9000'):
    """
    Constructor.

    :param (str) server: url of the CoreNLP server.
    """
    self.server = server
    assert requests.get(self.server).ok, 'Stanford CoreNLP server was not found at location {}'.format(self.server)

  def _request(self, text, properties):
    try:
      r = requests.post(self.server, params={'properties': str(properties)}, data=text)
      r.raise_for_status()
      return r
    except requests.HTTPError:
      raise RuntimeError(r.text)

  def annotate_dict(self, text, annotators):
    """Return a dict from the CoreNLP server, containing annotations of the text.

    :param (str) text: Text to annotate.
    :param (list[str]) annotators: a list of annotator names

    :return (dict): a dict of annotations
    """
    properties = {
      'annotators': ','.join(annotators),
      'outputFormat': 'json',
    }
    return self._request(text, properties).json(strict=False)

  def annotate_proto(self, text, annotators):
    """Return a Document protocol buffer from the CoreNLP server, containing annotations of the text.

    :param (str) text: text to be annotated
    :param (list[str]) annotators: a list of annotator names

    :return (CoreNLP_pb2.Document): a Document protocol buffer
    """
    properties = {
      'annotators': ','.join(annotators),
      'outputFormat': 'serialized',
      'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
    }
    r = self._request(text, properties)
    buffer = r.content  # bytes

    size, pos = _DecodeVarint(buffer, 0)
    buffer = buffer[pos:(pos + size)]
    doc = CoreNLP_pb2.Document()
    doc.ParseFromString(buffer)
    return doc

  def annotate(self, text, annotators):
    """Return an AnnotatedDocument from the CoreNLP server.

    :param (str) text: text to be annotated
    :param (list[str]) annotators: a list of annotator names

    See a list of valid annotator names here:
      http://stanfordnlp.github.io/CoreNLP/annotators.html

    :return (AnnotatedDocument): an annotated document
    """
    # TODO(kelvin): include raw text attribute for each sentence
    doc_pb = self.annotate_proto(text, annotators)
    return AnnotatedDocument(doc_pb)


class Document(Sequence):
  """A sequence of Sentence objects."""
  pass


class Sentence(Sequence):
  """A sequence of Token objects."""
  pass


class Token(object):
  @abstractproperty
  def word(self):
    pass


class AnnotatedDocument(Document):
  def __init__(self, doc_pb):
    self.pb = doc_pb
    self._sentences = [AnnotatedSentence(sent) for sent in self.pb.sentence]

  def __getitem__(self, i):
    return self._sentences[i]

  def __len__(self):
    return len(self._sentences)

  @staticmethod
  def from_dict(json_dict):
    return AnnotatedDocument(AnnotatedDocument.dict_to_pb(json_dict))

  @staticmethod
  def dict_to_pb(json_dict):
    sentences = [AnnotatedSentence.dict_to_pb(d) for d in json_dict['sentences']]
    doc = CoreNLP_pb2.Document()
    # TODO(kelvin): set doc.text and other available properties
    doc.sentence.extend(sentences)
    return doc

  @property
  def coref(self):
    raise NotImplementedError

  def __str__(self):
    return self.pb.text


# TODO(kelvin): finish specifying the Simple interface for AnnotatedSentence
# http://stanfordnlp.github.io/CoreNLP/simple.html
# In particular, all the methods that take arguments.


class AnnotatedSentence(Sentence):
  def __init__(self, sentence_pb):
    self.pb = sentence_pb
    self._tokens = [AnnotatedToken(tok) for tok in self.pb.token]

  def __getitem__(self, i):
    return self._tokens[i]

  def __len__(self):
    return len(self._tokens)

  def __str__(self):
    return self.text

  @staticmethod
  def dict_to_pb(json_dict):
    sent = CoreNLP_pb2.Sentence()
    tokens = [AnnotatedToken.dict_to_pb(d) for d in json_dict['tokens']]
    # TODO(kelvin): set other properties of sentence
    sent.token.extend(tokens)
    return sent

  def word(self, i):
    return self._tokens[i].word

  @property
  def words(self):
    return [tok.word for tok in self._tokens]

  @property
  def text(self):
    return self.pb.text

  def pos_tag(self, i):
    return self._tokens[i].pos

  @property
  def pos_tags(self):
    return [tok.pos for tok in self._tokens]

  def lemma(self, i):
    return self._tokens[i].lemma

  @property
  def lemmas(self):
    return [tok.lemma for tok in self._tokens]

  def ner_tag(self, i):
    return self._tokens[i].ner

  @property
  def ner_tags(self):
    return [tok.ner for tok in self._tokens]

  @property
  def parse(self):
    raise NotImplementedError

  @property
  def natlog_polarities(self):
    raise NotImplementedError

  @property
  def openie(self):
    raise NotImplementedError

  @property
  def openie_triples(self):
    raise NotImplementedError


class AnnotatedToken(Token):
  def __init__(self, token_pb):
    self.pb = token_pb

  @staticmethod
  def dict_to_pb(json_dict):
    tok = CoreNLP_pb2.Token()
    tok.word = json_dict['word']
    tok.pos = json_dict['pos']
    tok.ner = json_dict['ner']
    tok.lemma = json_dict['lemma']
    tok.wikipediaEntity = json_dict['entitylink']
    # TODO(kelvin): set other properties of token
    return tok

  @property
  def word(self):
    return self.pb.word

  @property
  def pos(self):
    return self.pb.pos

  @property
  def ner(self):
    return self.pb.ner

  @property
  def lemma(self):
    return self.pb.lemma

  @property
  def normalized_ner(self):
    return self.pb.normalizedNER

  @property
  def wikipedia_entity(self):
    return self.pb.wikipediaEntity


# TODO(kelvin): sentence and doc classes that lazily perform annotations
class LazyDocument(Sentence):
  pass


class LazySentence(Sentence):
  pass
