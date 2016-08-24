from abc import abstractmethod
from itertools import izip

import requests
from collections import defaultdict
from google.protobuf.internal.decoder import _DecodeVarint

from . import CoreNLP_pb2
from stanza.nlp.data import Document, Sentence, Token

__author__ = 'kelvinguu, vzhong, wmonroe4'


class AnnotationException(Exception):
    """
    Exception raised when there was an error communicating with the CoreNLP server.
    """
    pass


class CoreNLPClient(object):
    """
    A CoreNLP client to the Stanford CoreNLP server.
    """

    DEFAULT_ANNOTATORS = "tokenize ssplit lemma pos ner depparse".split()

    def __init__(self, server='http://localhost:9000', default_annotators=DEFAULT_ANNOTATORS):
        """
        Constructor.
        :param (str) server: url of the CoreNLP server.
        """
        self.server = server
        self.default_annotators = default_annotators
        assert requests.get(self.server).ok, 'Stanford CoreNLP server was not found at location {}'.format(self.server)

    def _request(self, text, properties):
        try:
            r = requests.post(self.server, params={'properties': str(properties)}, data=text.encode('utf-8'))
            r.raise_for_status()
            return r
        except requests.HTTPError:
            raise AnnotationException(r.text)

    def annotate_json(self, text, annotators=None):
        """Return a JSON dict from the CoreNLP server, containing annotations of the text.

        :param (str) text: Text to annotate.
        :param (list[str]) annotators: a list of annotator names

        :return (dict): a dict of annotations
        """
        properties = {
            'annotators': ','.join(annotators or self.default_annotators),
            'outputFormat': 'json',
        }
        return self._request(text, properties).json(strict=False)

    def annotate_proto(self, text, annotators=None):
        """Return a Document protocol buffer from the CoreNLP server, containing annotations of the text.

        :param (str) text: text to be annotated
        :param (list[str]) annotators: a list of annotator names

        :return (CoreNLP_pb2.Document): a Document protocol buffer
        """
        properties = {
            'annotators': ','.join(annotators or self.default_annotators),
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

    def annotate(self, text, annotators=None):
        """Return an AnnotatedDocument from the CoreNLP server.

        :param (str) text: text to be annotated
        :param (list[str]) annotators: a list of annotator names

        See a list of valid annotator names here:
          http://stanfordnlp.github.io/CoreNLP/annotators.html

        :return (AnnotatedDocument): an annotated document
        """
        doc_pb = self.annotate_proto(text, annotators)
        return AnnotatedDocument(doc_pb)


class ProtobufBacked(object):
    @property
    def pb(self):
        """Get the backing protocol buffer."""
        return self._pb

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.pb == other.pb

    def __ne__(self, other):
        return not self.__eq__(other)

    @abstractmethod
    def _from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Note: this should be a classmethod.
        """
        pass

    @classmethod
    def from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Ensures that the object has a protocol buffer property.
        """
        obj = cls._from_pb(pb)
        obj._pb = pb  # set the protocol buffer
        return obj

    @property
    def json(self):
        """The object represented as JSON.

        In the future, this should be computed from the protocol buffer. For now, it is manually set.
        """
        try:
            return self._json
        except AttributeError:
            raise AttributeError('No JSON representation available.')

    @json.setter
    def json(self, json_dict):
        self._json = json_dict

    def to_json(self):
        """Same as the json property.

        Provided just because people are accustomed to calling `to_json` on objects.
        """
        return self.json

    @abstractmethod
    def json_to_pb(cls, json_dict):
        """Convert JSON to protocol buffer.

        Note: This should be a classmethod.
        """
        pass

    @classmethod
    def from_json(cls, json_dict):
        pb = cls.json_to_pb(json_dict)
        obj = cls.from_pb(pb)
        obj._json = json_dict  # set the JSON


class AnnotatedDocument(Document, ProtobufBacked):
    """
    A shim over the protobuffer exposing key methods.
    """
    @classmethod
    def _from_pb(cls, pb):
        return cls(pb)

    def __init__(self, pb):
        self._sentences = [AnnotatedSentence.from_pb(sent_pb) for sent_pb in pb.sentence]

    def __getitem__(self, i):
        return self._sentences[i]

    def __len__(self):
        return len(self._sentences)

    def __str__(self):
        return self.pb.text

    def __repr__(self):
        PREVIEW_LEN = 50
        return "[Document: {}]".format(self.pb.text[:PREVIEW_LEN] + ("..." if len(self.pb.text) > PREVIEW_LEN else ""))

    @ProtobufBacked.json.setter
    def json(self, json_dict):
        self._json = json_dict
        # propagate JSON to children
        for sent, sent_json in izip(self._sentences, json_dict['sentences']):
            sent.json = sent_json

    @classmethod
    def json_to_pb(cls, json_dict):
        sentences = [AnnotatedSentence.json_to_pb(d) for d in json_dict['sentences']]
        doc = CoreNLP_pb2.Document()
        doc.sentence.extend(sentences)
        doc.text = AnnotatedDocument._reconstruct_text_from_sentence_pbs(sentences)
        return doc

    @staticmethod
    def _reconstruct_text_from_sentence_pbs(sentence_pbs):
        before = lambda sentence_pb: sentence_pb.token[0].before
        after = lambda sentence_pb: sentence_pb.token[-1].after

        text = []
        for i, sent in enumerate(sentence_pbs):
            if i == 0:
                text.append(before(sent))
            text.append(sent.text)
            text.append(after(sent))
        return ''.join(text)

    @property
    def doc_id(self):
        return self.pb.docID

    @property
    def text(self):
        return self.pb.text

    def __getattr__(self, attr):
        """
        If you are looking for an entry in the protobuf that hasn't been
        defined above, this will access it.
        """
        return getattr(self.pb, attr)

    @property
    def character_span(self):
        """
        Returns the character span of the sentence
        """
        return (self._sentences[0].character_span[0], self._sentences[-1].character_span[1])

    @property
    def sentences(self):
        return self._sentences

    # These are features that are yet to be supported. In the mean time,
    # users can struggle with the protobuf
    # @property
    # def coref(self):
    #     raise NotImplementedError()


# TODO(kelvin): finish specifying the Simple interface for AnnotatedSentence
# http://stanfordnlp.github.io/CoreNLP/simple.html
# In particular, all the methods that take arguments.

# TODO(kelvin): protocol buffers insert undesirable default values. Deal with these somehow.

class AnnotatedSentence(Sentence, ProtobufBacked):
    @classmethod
    def _from_pb(cls, pb):
        # Fill in the text attribute if needed.
        if len(pb.text) == 0:
            pb.text = cls._reconstruct_text_from_token_pbs(pb.token)
            print(pb.text)
        return cls(pb)

    @property
    def document(self):
        try:
            return self._document
        except AttributeError:
            raise AttributeError("Document has not been set.")

    @document.setter
    def document(self, val):
        self._document = val

    @classmethod
    def _reconstruct_text_from_token_pbs(cls, token_pbs):
        text = []
        tok = None
        for i, tok in enumerate(token_pbs):
            if i != 0:
                text.append(tok.before)
            text.append(tok.word)
        return ''.join(text)

    def __init__(self, pb):
        self._tokens = [AnnotatedToken.from_pb(tok_pb) for tok_pb in pb.token]

    @ProtobufBacked.json.setter
    def json(self, json_dict):
        self._json = json_dict
        # propagate JSON to children
        for tok, tok_json in izip(self._tokens, json_dict['tokens']):
            tok.json = tok_json

    def __getitem__(self, i):
        return self._tokens[i]

    def __len__(self):
        return len(self._tokens)

    def __str__(self):
        return self.text.encode('utf-8')

    def __unicode__(self):
        return self.text

    def __repr__(self):
        PREVIEW_LEN = 50
        return "[Sentence: {}]".format(self.text[:PREVIEW_LEN] + ("..." if len(self.pb.text) > PREVIEW_LEN else ""))

    @classmethod
    def json_to_pb(cls, json_dict):
        sent = CoreNLP_pb2.Sentence()
        tokens = [AnnotatedToken.json_to_pb(d) for d in json_dict['tokens']]
        sent.token.extend(tokens)
        return sent

    @property
    def paragraph(self):
        """
        Returns the paragraph index.
        """
        return self.pb.paragraph

    @property
    def sentenceIndex(self):
        """
        Returns the paragraph index.
        """
        return self.pb.sentenceIndex

    def next_sentence(self):
        """
        Returns the next sentence
        """
        return self.document[self.sentenceIndex + 1]

    def previous_sentence(self):
        """
        Returns the previous sentence
        """
        return self.document[self.sentenceIndex - 1]

    def word(self, i):
        return self._tokens[i].word

    @property
    def before(self):
        return self._tokens[0].before

    @property
    def after(self):
        return self._tokens[-1].after

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
    def tokens(self):
        return self._tokens

    def token(self, i):
        return self._tokens[i]

    @property
    def depparse(self, mode="enhancedPlusPlus"):
        """
        Retrieves the appropriate dependency parse.
        Must be one of:
            - basic
            - alternative
            - collapsedCCProcessed
            - collapsed
            - enhanced
            - enhancedPlusPlus
        """
        assert mode in [
            "basic",
            "alternative",
            "collapsedCCProcessed",
            "collapsed",
            "enhanced",
            "enhancedPlusPlus", ], "Invalid mode"
        dep_pb = getattr(self.pb, mode + "Dependencies")
        if dep_pb is None:
            raise AttributeError("No dependencies for mode: " + mode)
        else:
            return AnnotatedDependencyParseTree(dep_pb, self)

    @property
    def character_span(self):
        """
        Returns the character span of the sentence
        """
        return (self._tokens[0].character_span[0], self._tokens[-1].character_span[1])

    def __getattr__(self, attr):
        return getattr(self.pb, attr)

    # @property
    # def parse(self):
    #    raise NotImplementedError()

    # @property
    # def natlog_polarities(self):
    #    raise NotImplementedError

    # @property
    # def relations(self, mode="kbp"):
    #    """
    #    Returns any relations found by the annotators.
    #    Valid modes are:
    #        - kbp
    #        - openie
    #        - relation (?)
    #    """
    #    raise NotImplementedError()

    # @property
    # def openie(self):
    #    raise NotImplementedError

    # @property
    # def openie_triples(self):
    #    raise NotImplementedError

    # @property
    # def mentions(self):
    #    """
    #    Supposed to return mentions contained in the sentence.
    #    """
    #    raise NotImplementedError


class AnnotatedToken(Token, ProtobufBacked):
    @classmethod
    def _from_pb(cls, pb):
        return cls(pb)

    def __init__(self, token_pb):
        pass

    def __str__(self):
        return self.pb.word

    def __repr__(self):
        return "[Token: {}]".format(self.pb.word)

    @classmethod
    def json_to_pb(cls, json_dict):
        tok = CoreNLP_pb2.Token()

        def assign_if_present(pb_key, dict_key):
            if dict_key in json_dict:
                setattr(tok, pb_key, json_dict[dict_key])

        mapping = {
            'after': 'after',
            'before': 'before',
            'beginChar': 'characterOffsetBegin',
            'endChar': 'characterOffsetEnd',
            'originalText': 'originalText',
            'word': 'word',
            'pos': 'pos',
            'ner': 'ner',
            'lemma': 'lemma',
            'wikipediaEntity': 'entitylink',
        }

        for pb_key, dict_key in mapping.items():
            assign_if_present(pb_key, dict_key)

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
    def originalText(self):
        return self.pb.originalText

    @property
    def before(self):
        return self.pb.before

    @property
    def after(self):
        return self.pb.after

    @property
    def normalized_ner(self):
        return self.pb.normalizedNER

    @property
    def wikipedia_entity(self):
        return self.pb.wikipediaEntity

    @property
    def character_span(self):
        """
        Returns the character span of the token
        """
        return (self.pb.beginChar, self.pb.endChar)


class AnnotatedDependencyParseTree():
    """
    Represents a dependency parse tree
    """

    def __init__(self, pb, sentence=None):
        self.pb = pb
        self.sentence = sentence
        self.graph, self.inv_graph = AnnotatedDependencyParseTree.parse_graph(pb.edge)

    @staticmethod
    def parse_graph(edges):
        graph = defaultdict(list)
        inv_graph = defaultdict(list)
        for edge in edges:
            graph[edge.source].append((edge.target, edge.dep))
            inv_graph[edge.target].append((edge.source, edge.dep))

        return graph, inv_graph

    @property
    def roots(self):
        return self.pb.root

    @property
    def parents(self, i):
        return self.inv_graph[i]

    @property
    def children(self, i):
        return self.graph[i]


# TODO(kelvin): sentence and doc classes that lazily perform annotations
class LazyDocument(Sentence):
    pass


class LazySentence(Sentence):
    pass
