import importlib
import sys
import warnings

warnings.warn('pick_gpu has been moved from stanza.research to stanza.cluster; the module in research is deprecated.')

sys.modules[__name__] = importlib.import_module('...cluster.pick_gpu', __name__)
