import numpy as np

from .instance import Instance  # NOQA: for doctest
from .learner import Learner  # NOQA: for doctest


def log_likelihood(eval_data, predictions, scores, learner='ignored'):
    '''
    Return the log likelihood of each correct output, which is simply equal to
    the score in `scores`.

    >>> log_likelihood(None, None, [-0.5, -1.0, -2.0])
    [-0.5, -1.0, -2.0]
    '''
    return scores


def log_likelihood_bits(eval_data, predictions, scores, learner='ignored'):
    '''
    Return the log likelihood of each correct output in base 2 (bits),
    computed from the scores in `scores` (which should be in base e, nats).

    >>> bits = log_likelihood_bits(None, None, [np.log(0.5), np.log(0.125), np.log(0.25)])
    >>> [round(b) for b in bits]
    [-1.0, -3.0, -2.0]
    '''
    return (np.array(scores) / np.log(2.0)).tolist()


def accuracy(eval_data, predictions, scores='ignored', learner='ignored'):
    '''
    Return the accuracy of each prediction in `predictions`: 1 if it is equal
    to the correct output in `eval_data`, 0 otherwise.

    >>> data = [Instance('input', 'correct'),
    ...         Instance('input', 'correct'),
    ...         Instance('input', 'correct')]
    >>> accuracy(data, ['correct', 'wrong', 'correct'])
    [1, 0, 1]
    '''
    return [int(inst.output == pred)
            for inst, pred in zip(eval_data, predictions)]


def squared_error(eval_data, predictions, scores='ignored', learner='ignored'):
    '''
    Return the squared error of each prediction in `predictions` with respect
    to the correct output in `eval_data`.

    >>> data = [Instance('input', (0., 0., 1.)),
    ...         Instance('input', (0., 1., 1.)),
    ...         Instance('input', (1., 0., 0.))]
    >>> squared_error(data, [(0., 1., 1.), (0., 1., 1.), (-1., 1., 0.)])
    [1.0, 0.0, 5.0]
    '''
    return [np.sum((np.array(pred) - np.array(inst.output)) ** 2)
            for inst, pred in zip(eval_data, predictions)]


def perplexity(eval_data, predictions, scores, learner='ignored'):
    '''
    Return the perplexity `exp(-score)` computed from each score in `scores`.
    The log scores in `scores` should be base e (`exp`, `log`).

    The correct average to use for this metric is the geometric mean. It is
    recommended to work in log space to calcuate this mean (or use
    `scipy.stats.mstats.gmean`):

        mean_perplexity = np.exp(np.log(perplexities).mean())

    >>> perplexities = perplexity(None, None, [np.log(0.5), np.log(0.1), np.log(0.25)])
    >>> [round(p) for p in perplexities]
    [2.0, 10.0, 4.0]
    '''
    return np.exp(-np.array(scores)).tolist()


def aic(eval_data, predictions, scores, learner):
    '''
    Return Akaike information criterion (AIC) scores for the given
    `learner` producing the given `scores` (log likelihoods in base e):

        aic = 2 * learner.num_params - 2 * sum(log_2(exp(scores)))

    The result is a list *one element longer* than the number of scores:
    the last element of this list is the penalty for the learner from the
    number of parameters, and the others are negative log likelihoods in
    base 2.

    The standard way to aggregate this metric is to sum the resulting list.

    >>> learner = Learner(); learner.num_params = 1024
    >>> aic(None, None, [np.log(0.5), np.log(0.125), np.log(0.25)], learner)
    [2.0, 6.0, 4.0, 2048.0]
    '''
    return (-2.0 * np.array(scores) / np.log(2.0)).tolist() + [2.0 * float(learner.num_params)]
