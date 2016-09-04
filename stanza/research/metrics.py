import numpy as np
'''
import warnings
try:
    import nltk.translate.bleu_score as nltk_bleu
except ImportError as e:
    warnings.warn('Cannot import nltk; BLEU will be unavailable: ' + str(e))
    nltk_bleu = None
'''

from .bleu import corpus_bleu
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


def prec1(eval_data, predictions, scores='ignored', learner='ignored'):
    '''
    Return the precision@1 of each prediction in `predictions`: 1 if it is equal
    to any of the correct outputs for the corresponding instance in `eval_data`,
    0 otherwise.

    >>> data = [Instance('input', ['correct', 'right']),
    ...         Instance('input', ['correct', 'right']),
    ...         Instance('input', ['correct', 'right'])]
    >>> prec1(data, ['correct', 'wrong', 'right'])
    [1, 0, 1]
    '''
    return [int(any(o == pred for o in inst.output))
            for inst, pred in zip(eval_data, predictions)]


def bleu(eval_data, predictions, scores='ignored', learner='ignored'):
    '''
    Return corpus-level BLEU score of `predictions` using the `output`
    field of the instances in `eval_data` as references. This is returned
    as a length-1 list of floats.

    This uses the NLTK unsmoothed implementation, which has been known
    to have some bugs. This function patches over the biggest bug, which
    is that NLTK ignores n-gram overlap counts of 0 (this should result
    in a zero BLEU score).

    >>> data = [Instance('input', 'this is the good'),
    ...         Instance('input', 'the bad'),
    ...         Instance('input', 'and the ugly')]
    >>> bleu(data, ['this is the good', 'the good', 'seriously really good'])  # doctest: +ELLIPSIS
    [0.65599...]
    >>> np.exp(np.mean([np.log(5. / 9.), np.log(3. / 6.),
    ...                 np.log(2. / 3.), np.log(1. / 1.)]))  # doctest: +ELLIPSIS
    0.65599...
    '''
    ref_groups = ([inst.output.split()]
                  if isinstance(inst.output, basestring) else
                  [r.split() for r in inst.output]
                  for inst in eval_data)
    return [corpus_bleu(ref_groups, [p.split() for p in predictions])]


def _has_4gram_match(ref, pred):
    '''
    >>> _has_4gram_match(['four', 'lovely', 'tokens', 'here'],
    ...                  ['four', 'lovely', 'tokens', 'here'])
    True
    >>> _has_4gram_match(['four', 'lovely', 'tokens', 'here'],
    ...                  ['four', 'lovely', 'tokens', 'here', 'and', 'there'])
    True
    >>> _has_4gram_match(['four', 'lovely', 'tokens', 'here'],
    ...                  ['four', 'ugly', 'tokens', 'here'])
    False
    >>> _has_4gram_match(['four', 'lovely', 'tokens'],
    ...                  ['lovely', 'tokens', 'here'])
    False
    '''
    if len(ref) < 4 or len(pred) < 4:
        return False

    for i in range(len(ref) - 3):
        for j in range(len(pred) - 3):
            if ref[i:i + 4] == pred[j:j + 4]:
                return True
    return False


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


def token_perplexity_macro(eval_data, predictions, scores, learner='ignored'):
    '''
    Return the per-token perplexity `exp(-score / num_tokens)` computed from each
    score in `scores.`

    The correct macro-average is given by the geometric mean.

    >>> refs = [Instance(None, ''),
    ...         Instance(None, ''),
    ...         Instance(None, '2')]
    >>> scores = [np.log(1.0), np.log(0.25), np.log(1 / 64.)]
    >>> perplexities = token_perplexity_macro(refs, None, scores)
    >>> [round(p) for p in perplexities]
    ... # sequence perplexities: [1, 4, 16]
    ... # per-token perplexities: [1, 4, 8]
    [1.0, 4.0, 8.0]
    '''
    lens = np.array([len(inst.output.split()) + 1 for inst in eval_data])
    return np.exp(-np.array(scores) / lens).tolist()


def token_perplexity_micro(eval_data, predictions, scores, learner='ignored'):
    '''
    Return the micro-averaged per-token perplexity `exp(-score / num_tokens)`
    computed over the entire corpus, as a length-1 list of floats.
    The log scores in `scores` should be base e (`exp`, `log`).

    >>> refs = [Instance(None, ''),
    ...         Instance(None, ''),
    ...         Instance(None, '2')]
    >>> scores = [np.log(1.0), np.log(0.25), np.log(1 / 64.)]
    >>> perplexity = token_perplexity_micro(refs, None, scores)
    >>> [round(p) for p in perplexity]
    ... # sequence perplexities: [1, 4, 64]
    ... # per-token perplexities: [1, 4, 8]
    ... # micro-average: gmean([1, 4, 8, 8])
    [4.0]
    '''
    lens = np.array([len(inst.output.split()) + 1 for inst in eval_data])
    return [np.exp(np.average(-np.array(scores) / lens, weights=lens))]


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
    >>> aic(None, None, [np.log(0.5), np.log(0.125), np.log(0.25), np.log(0.5)], learner)
    [2.0, 6.0, 4.0, 2.0, 2048.0]
    '''
    return (-2.0 * np.array(scores) / np.log(2.0)).tolist() + [2.0 * float(learner.num_params)]


def aic_averaged(eval_data, predictions, scores, learner):
    '''
    Return Akaike information criterion (AIC) scores for the given
    `learner` producing the given `scores` (log likelihoods in base e):

        aic = 2 * learner.num_params - 2 * sum(log_2(exp(scores)))

    The result is a list of the same length as the number of scores.
    The penalty from the number of parameters is divided by the number of
    scores and added to the contribution of each score; thus, `aic` and
    `aic_averaged` will have the same mean but yield different-size lists.

    The standard way to aggregate this metric is to sum the resulting list.

    >>> learner = Learner(); learner.num_params = 1024
    >>> aic_averaged(None, None, [np.log(0.5), np.log(0.125), np.log(0.25), np.log(0.5)], learner)
    [514.0, 518.0, 516.0, 514.0]
    '''
    scores = np.array(scores)
    penalty = 2.0 * float(learner.num_params) / len(scores)
    return (penalty - 2.0 * scores / np.log(2.0)).tolist()


METRICS = {
    name: globals()[name]
    for name in dir()
    if (name not in ['np', 'corpus_bleu', 'Instance', 'Learner']
        and not name.startswith('_'))
}
