import importlib
import sys
import warnings

warnings.warn('stanza.unstable has been renamed to stanza.research; the name unstable is deprecated.')

sys.modules[__name__] = importlib.import_module('..research', __name__)
