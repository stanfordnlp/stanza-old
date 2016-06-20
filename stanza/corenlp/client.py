import requests

__author__ = 'vzhong'

# Original work by Smitha Milli:
# https://github.com/smilli/py-corenlp



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
