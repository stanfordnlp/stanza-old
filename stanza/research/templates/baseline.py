# -*- coding: utf-8 -*-
import argparse

from stanza.monitoring import progress
from stanza.research import config
from stanza.research.learner import Learner


# TODO: name of model
class BaselineLearner(Learner):
    def __init__(self):
        self.get_options()
        # TODO: initialize parameters

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        # TODO: train model
        pass

    @property
    def num_params(self):
        total = 0
        # TODO: count parameters
        return total

    def predict_and_score(self, eval_instances, random='ignored', verbosity=4):
        eval_instances = list(eval_instances)
        predictions = []
        scores = []

        if verbosity >= 1:
            progress.start_task('Eval instance', len(eval_instances))

        for i, inst in enumerate(eval_instances):
            if verbosity >= 1:
                progress.progress(i)

            pred = ''  # TODO: make prediction
            score = -float('inf')  # TODO: score gold output
            predictions.append(pred)
            scores.append(score)

        if verbosity >= 1:
            progress.end_task()

        return predictions, scores

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)
