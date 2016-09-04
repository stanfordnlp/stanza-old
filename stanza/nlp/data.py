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

class Entity(object):
    """An 'entity' in a information extraction sense. Each entity has
    a type, a token sequence in a sentence and an optional canonical
    link (if coreference is present). """

    @abstractproperty
    def sentence(self):
        """Returns the referring sentence"""
        pass

    @abstractproperty
    def head_token(self):
        """Returns the start token."""
        pass

    @abstractproperty
    def token_span(self):
        """Returns the index of the end token."""
        pass

    @abstractproperty
    def character_span(self):
        """Returns the index of the end character."""
        pass

    @abstractproperty
    def type(self):
        """Returns the type of the string"""
        pass

    @abstractproperty
    def gloss(self):
        """Returns the exact string of the entity"""
        pass

    @abstractproperty
    def canonical_entity(self):
        """Returns the exact string of the canonical reference"""
        pass

