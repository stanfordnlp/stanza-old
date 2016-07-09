from abc import ABCMeta, abstractmethod
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

    :param server: url of the CoreNLP server.
    """
    self.server = server
    assert requests.get(self.server).ok, 'Stanford CoreNLP server was not found at location {}'.format(self.server)

  def annotate(self, text, properties=None):
    """
    Annotates text using CoreNLP. The properties field are described in

    http://stanfordnlp.github.io/CoreNLP/corenlp-server.html.

    The properties for each of the annotators can be found at

    http://stanfordnlp.github.io/CoreNLP/annotators.html

    :param text: Text to annotate.
    :param properties: A dictionary of properties for CoreNLP.
    """
    properties = properties or {}
    r = requests.get(self.server, params={'properties': str(properties)}, data=text)
    return r.json(strict=False)

class Document(Sequence):
  """A sequence of Sentences."""
  pass


class Sentence(Sequence):
  """A sequence of strings."""
  pass


class AnnotatedDocument(Document):
  def __init__(self, doc_pb):
    self.pb = doc_pb

  def __getitem__(self, i):
    return AnnotatedSentence(self.pb.sentence[i])

  def __len__(self):
    return len(self.pb.sentence)

  @property
  def coref(self):
      # TODO
      raise NotImplementedError

  def __str__(self):
    return self.pb.text


# TODO: finish specifying the Simple interface for AnnotatedSentence
# http://stanfordnlp.github.io/CoreNLP/simple.html
# In particular, all the methods that take arguments.


class AnnotatedSentence(Sentence):
  def __init__(self, sentence_pb):
    self.pb = sentence_pb

  def __getitem__(self, i):
    return self.pb.token[i].word

  def __len__(self):
    return len(self.pb.token)

  def __str__(self):
    return self.text

  @property
  def text(self):
    return self.pb.text

  @property
  def pos_tags(self):
    raise NotImplementedError

  @property
  def lemmas(self):
    raise NotImplementedError

  @property
  def ner_tags(self):
    raise NotImplementedError

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