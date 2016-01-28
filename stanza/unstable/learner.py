import cPickle as pickle


class Learner(object):
    def __init__(self):
        self._using_default_separate = False
        self._using_default_combined = False

    def train(self, training_instances):
        '''
        Fit a model on training data.

        :param training_instances: The data to use to train the model.
            Instances should have at least the `input` and `output` fields
            populated.
        :type training_instances: list(instance.Instance)

        :returns: None
        '''
        raise NotImplementedError

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
        if self._using_default_combined:
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
        if self._using_default_combined:
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
        if self._using_default_separate:
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
