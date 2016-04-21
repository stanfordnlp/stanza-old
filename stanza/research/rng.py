import numpy as np

from . import config

parser = config.get_options_parser()
parser.add_argument('--random_seed', default='DefaultRandomSeed',
                    help='A string for initializing the random number generator, '
                         'for reproducible experiments. The string will be hashed '
                         "and the hash used as the seed to numpy's RandomState.")

_random_state = None


def get_rng():
    global _random_state
    if _random_state is None:
        options, _ = parser.parse_known_args()
        _random_state = np.random.RandomState(np.uint32(hash(options.random_seed)))

    return _random_state
