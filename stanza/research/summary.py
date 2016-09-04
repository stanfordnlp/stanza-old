import importlib
import sys
import warnings

warnings.warn('summary has been moved from stanza.research to stanza.monitoring; the module in research is deprecated.')

sys.modules[__name__] = importlib.import_module('...monitoring.summary', __name__)
