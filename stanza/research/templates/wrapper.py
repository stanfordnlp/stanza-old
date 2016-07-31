import cPickle as pickle

from stanza.research.instance import Instance

DEFAULT_MODEL = 'models/__INSERT_MODEL_HERE__/model.pkl'


class Wrapper(object):
    '''
    A wrapper class for pickled Learners.
    '''
    def __init__(self, picklefile=None):
        '''
        :param file picklefile: An open file-like object from which to
            load the model. Can be produced either from a normal experiment
            run or a quickpickle.py run. If `None`, try to load the default
            quickpickle file (this is less future-proof than the normal
            experiment-produced pickle files).
        '''
        if picklefile is None:
            with open(DEFAULT_MODEL, 'rb') as infile:
                self.model = pickle.load(infile)
        else:
            self.model = pickle.load(picklefile)
        self.model.options.verbosity = 0

    def process(self, input):
        return self.process_all([input])[0]

    def process_all(self, inputs):
        insts = [Instance(i) for i in inputs]
        return self.model.predict(insts, verbosity=0)
