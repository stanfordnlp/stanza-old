class Instance(object):
    '''
    Represents an individual data point in a training or testing set, for a classifier
    trained to predict `output` given `input`.

    `alt_inputs` and `alt_outputs` are optional lists of alternative stimuli and
    predictions, respectively, for use in pragmatic settings. `annotated_input` and
    `annotated_output` can be used for versions of the input and output that have
    been augmented with additional data, e.g. parse trees, logical forms, POS tags.

    `source` is the original object from which this instance is constructed.
    '''
    def __init__(self,
                 input, output=None,
                 annotated_input=None, annotated_output=None,
                 alt_inputs=None, alt_outputs=None,
                 source=None):
        self.source = source
        self.input, self.output = input, output
        self.annotated_input, self.annotated_output = annotated_input, annotated_output
        self.alt_inputs, self.alt_outputs = alt_inputs, alt_outputs

    def stripped(self, include_annotated=True):
        '''
        Return a version of this instance with all information removed that could be used
        to "cheat" at test time: the true output and its annotated version, and the
        reference to the full source.

        If `include_annotated` is true, `annotated_input` will also be included (but not
        `annotated_output` in either case).
        '''
        return Instance(self.input,
                        annotated_input=self.annotated_input if include_annotated else None,
                        alt_inputs=self.alt_inputs, alt_outputs=self.alt_outputs)

    def inverted(self):
        '''
        Return a version of this instance with inputs replaced by outputs and vice versa.
        '''
        return Instance(input=self.output, output=self.input,
                        annotated_input=self.annotated_output,
                        annotated_output=self.annotated_input,
                        alt_inputs=self.alt_outputs,
                        alt_outputs=self.alt_inputs,
                        source=self.source)

    def __repr__(self):
        return 'Instance(%s, %s)' % (repr(self.input), repr(self.output))
