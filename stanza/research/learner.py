import cPickle as pickle

from . import evaluate, output


class Learner(object):
    def __init__(self):
        self._using_default_separate = False
        self._using_default_combined = False

    def train(self, training_instances, validation_instances=None, metrics=None):
        '''
        Fit a model on training data.

        :param training_instances: The data to use to train the model.
            Instances should have at least the `input` and `output` fields
            populated.
        :type training_instances: list(instance.Instance)

        :param validation_instances: The data to use to validate the model.
            Good practice says this should be held out (separate from the
            training set), but this API does not require that to be the case.
        :type validation_instances: list(instance.Instance)

        :param metrics: Functions like those found in the `metrics` module
            to use in validation. (These are not necessarily the objective function
            for training; subclasses define their own training objectives.)
        :type metrics: list(function)

        :returns: None
        '''
        raise NotImplementedError

    def validate(self, validation_instances, metrics, iteration=None):
        '''
        Evaluate this model on `validation_instances` during training and
        output a report.

        :param validation_instances: The data to use to validate the model.
        :type validation_instances: list(instance.Instance)

        :param metrics: Functions like those found in the `metrics` module
            for quantifying the performance of the learner.
        :type metrics: list(function)

        :param iteration: A label (anything with a sensible `str()` conversion)
            identifying the current iteration in output.
        '''
        if not validation_instances or not metrics:
            return {}
        split_id = 'val%s' % iteration if iteration is not None else 'val'
        train_results = evaluate.evaluate(self, validation_instances,
                                          metrics=metrics, split_id=split_id)
        output.output_results(train_results, split_id)
        return train_results

    def predict(self, eval_instances, random=False, verbosity=0):
        '''
        Return most likely predictions for each testing instance in
        `eval_instances`.

        :param eval_instances: The data to use to evaluate the model.
            Instances should have at least the `input` field populated.
            The `output` field need not be populated; subclasses should
            ignore it if it is present.
        :param random: If `True`, sample from the probability distribution
            defined by the classifier rather than output the most likely
            prediction.
        :param verbosity: The level of diagnostic output, relative to the
            global --verbosity option. Used to adjust output when models
            are composed of multiple sub-models.
        :type eval_instances: list(instance.Instance)

        :returns: list(output_type)
        '''
        if hasattr(self, '_using_default_combined') and self._using_default_combined:
            raise NotImplementedError

        self._using_default_separate = True
        return self.predict_and_score(eval_instances, random=random, verbosity=verbosity)[0]

    def score(self, eval_instances, verbosity=0):
        '''
        Return scores (negative log likelihoods) assigned to each testing
        instance in `eval_instances`.

        :param eval_instances: The data to use to evaluate the model.
            Instances should have at least the `input` and `output` fields
            populated. `output` is needed to define which score is to
            be returned.
        :param verbosity: The level of diagnostic output, relative to the
            global --verbosity option. Used to adjust output when models
            are composed of multiple sub-models.
        :type eval_instances: list(instance.Instance)

        :returns: list(float)
        '''
        if hasattr(self, '_using_default_combined') and self._using_default_combined:
            raise NotImplementedError

        self._using_default_separate = True
        return self.predict_and_score(eval_instances, verbosity=verbosity)[1]

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        '''
        Return most likely outputs and scores for the particular set of
        outputs given in `eval_instances`, as a tuple. Return value should
        be equivalent to the default implementation of

            return (self.predict(eval_instances), self.score(eval_instances))

        but subclasses can override this to combine the two calls and reduce
        duplicated work. Either the two separate methods or this one (or all
        of them) should be overridden.

        :param eval_instances: The data to use to evaluate the model.
            Instances should have at least the `input` and `output` fields
            populated. `output` is needed to define which score is to
            be returned.
        :param random: If `True`, sample from the probability distribution
            defined by the classifier rather than output the most likely
            prediction.
        :param verbosity: The level of diagnostic output, relative to the
            global --verbosity option. Used to adjust output when models
            are composed of multiple sub-models.
        :type eval_instances: list(instance.Instance)

        :returns: tuple(list(output_type), list(float))
        '''
        if hasattr(self, '_using_default_separate') and self._using_default_separate:
            raise NotImplementedError

        self._using_default_combined = True
        return (self.predict(eval_instances, random=random, verbosity=verbosity),
                self.score(eval_instances, verbosity=verbosity))

    def dump(self, outfile):
        '''
        Serialize the model for this learner and write it to a file.
        Serialized models can be loaded back in with `load`.

        By default, pickle the entire object. This may not be very efficient
        or reliable for long-term storage; consider overriding this (and `load`)
        to serialize only the necessary parameters. Alternatively, you can
        define __getstate__ and __setstate__ for subclasses to influence how
        the model is pickled (see https://docs.python.org/2/library/pickle.html).

        :param file outfile: A file-like object where the serialized model will
            be written.
        '''
        pickle.dump(self, outfile)

    def load(self, infile):
        '''
        Deserialize a model from a stored file.

        By default, unpickle an entire object. If `dump` is overridden to
        use a different storage format, `load` should be as well.

        :param file outfile: A file-like object from which to retrieve the
            serialized model.
        '''
        model = pickle.load(infile)
        self.__dict__.update(model.__dict__)
