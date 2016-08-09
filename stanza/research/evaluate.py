import numpy as np
import warnings

from . import config


def evaluate(learner, eval_data, metrics, metric_names=None, split_id=None,
             write_data=False):
    '''
    Evaluate `learner` on the instances in `eval_data` according to each
    metric in `metric`, and return a dictionary summarizing the values of
    the metrics.

    Dump the predictions, scores, and metric summaries in JSON format
    to "{predictions|scores|results}.`split_id`.json" in the run directory.

    :param learner: The model to be evaluated.
    :type learner: learner.Learner

    :param eval_data: The data to use to evaluate the model.
    :type eval_data: list(instance.Instance)

    :param metrics: An iterable of functions that defines the standard by
        which predictions are evaluated.
    :type metrics: Iterable(function(eval_data: list(instance.Instance),
                                     predictions: list(output_type),
                                     scores: list(float)) -> list(float))

    :param bool write_data: If `True`, write out the instances in `eval_data`
        as JSON, one per line, to the file `data.<split_id>.jsons`.
    '''
    if metric_names is None:
        metric_names = [
            (metric.__name__ if hasattr(metric, '__name__')
             else ('m%d' % i))
            for i, metric in enumerate(metrics)
        ]

    split_prefix = split_id + '.' if split_id else ''

    if write_data:
        config.dump([inst.__dict__ for inst in eval_data],
                    'data.%sjsons' % split_prefix,
                    default=lambda o: o.__dict__, lines=True)

    results = {split_prefix + 'num_params': learner.num_params}

    predictions, scores = learner.predict_and_score(eval_data)
    config.dump(predictions, 'predictions.%sjsons' % split_prefix, lines=True)
    config.dump(scores, 'scores.%sjsons' % split_prefix, lines=True)

    for metric, metric_name in zip(metrics, metric_names):
        prefix = split_prefix + (metric_name + '.' if metric_name else '')

        inst_outputs = metric(eval_data, predictions, scores, learner)
        if metric_name in ['data', 'predictions', 'scores']:
            warnings.warn('not outputting metric scores for metric "%s" because it would shadow '
                          'another results file')
        else:
            config.dump(inst_outputs, '%s.%sjsons' % (metric_name, split_prefix), lines=True)

        mean = np.mean(inst_outputs)
        gmean = np.exp(np.log(inst_outputs).mean())
        sum = np.sum(inst_outputs)
        std = np.std(inst_outputs)

        results.update({
            prefix + 'mean': mean,
            prefix + 'gmean': gmean,
            prefix + 'sum': sum,
            prefix + 'std': std,
            # prefix + 'ci_lower': ci_lower,
            # prefix + 'ci_upper': ci_upper,
        })

    config.dump_pretty(results, 'results.%sjson' % split_prefix)

    return results
