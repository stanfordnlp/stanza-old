import importlib
import sys
import warnings

warnings.warn('progress has been moved from stanza.research to stanza.monitoring; the module in research is deprecated.')

sys.modules[__name__] = importlib.import_module('...monitoring.progress', __name__)
