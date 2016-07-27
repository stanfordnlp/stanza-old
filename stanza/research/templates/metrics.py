from stanza.research.metrics import *


# TODO: define new metrics


METRICS = {
    name: globals()[name]
    for name in dir()
    if (name not in ['np']
        and not name.startswith('_'))
}
