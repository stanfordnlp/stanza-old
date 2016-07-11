from abc import abstractproperty
from collections import Sequence

__author__ = 'kelvinguu'


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