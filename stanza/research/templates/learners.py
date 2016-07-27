# TODO: import all learner models
from baseline import BaselineLearner


def new(key):
    '''
    Construct a new learner with the class named by `key`. A list
    of available learners is in the dictionary `LEARNERS`.
    '''
    return LEARNERS[key]()


LEARNERS = {
    'Baseline': BaselineLearner,
}
