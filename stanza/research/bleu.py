'''
An implementation of unsmoothed corpus-level BLEU.
'''

__author__ = 'wmonroe4'

from math import log, exp
from collections import Counter


def corpus_bleu(reference_groups, predictions):
    nums = [0] * 4
    denoms = [0] * 4
    prediction_len = 0
    reference_len = 0

    for refs, pred in zip(reference_groups, predictions):
        for n in range(1, 5):
            correct, total = modified_ngram_precision(refs, pred, n)
            nums[n - 1] += correct
            denoms[n - 1] += total

        prediction_len += len(pred)
        reference_len += closest_length(refs, pred)

    if prediction_len <= 0:
        brevity_penalty = 0.0 if reference_len > 0 else 1.0
    else:
        brevity_penalty = min(1.0, exp(1.0 - reference_len * 1.0 / prediction_len))
    fracs = [(num, denom) for num, denom in zip(nums, denoms) if denom != 0]
    if not fracs or 0 in [num for num, denom in fracs]:
        return 0.0
    else:
        weight = 1.0 / len(fracs)
        geom_mean = exp(sum(weight * log(num * 1.0 / denom) for num, denom in fracs))
        return brevity_penalty * geom_mean


def modified_ngram_precision(references, pred, n):
    '''
    Borrowed from the ntlk BLEU implementation:
    http://www.nltk.org/_modules/nltk/translate/bleu_score.html

    >>> modified_ngram_precision([['the', 'fat', 'cat', 'the', 'rat']],
    ...                          ['the', 'the', 'the', 'the', 'the'], 1)
    (2, 5)
    >>> modified_ngram_precision([['the', 'fat', 'cat', 'the', 'rat']],
    ...                          ['the', 'fat', 'the', 'rat'], 2)
    (2, 3)
    '''
    counts = Counter(iter_ngrams(pred, n))
    max_counts = {}
    for reference in references:
        reference_counts = Counter(iter_ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])

    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    denominator = sum(counts.values())
    return numerator, denominator


def iter_ngrams(s, n):
    return (tuple(s[i:i + n]) for i in range(len(s) - n + 1))


def closest_length(refs, pred):
    '''
    >>> closest_length(['1234', '12345', '1'], '123')
    4
    >>> closest_length(['123', '12345', '1'], '12')
    1
    '''
    smallest_diff = float('inf')
    closest_length = float('inf')
    for ref in refs:
        diff = abs(len(ref) - len(pred))
        if diff < smallest_diff or (diff == smallest_diff and len(ref) < closest_length):
            smallest_diff = diff
            closest_length = len(ref)
    return closest_length


def corpus_wer(reference_groups, predictions):
    import Levenshtein as lev
    errors = 0
    reference_len = 0
    for refs, pred in zip(reference_groups, predictions):
        coded = _code_tokens_to_unicode(refs + [pred])
        refs_coded = coded[:-1]
        pred_coded = coded[-1]
        dist, neg_len = min((lev.distance(ref, pred_coded), -len(ref)) for ref in refs_coded)
        errors += dist
        reference_len += -neg_len
    return errors * 1.0 / reference_len


def _code_tokens_to_unicode(seqs):
    r'''
    Converts a collection of tokenized sequences (which can be any hashable object)
    to Unicode strings, with each token type mapped to a sequentially increasing
    codepoint. This is to allow the use of string algorithms like Levenshtein on
    token sequences.

    >>> _code_tokens_to_unicode([['the', 'fat', 'cat'], ['cat', 'cafe']])
    [u'\x01\x02\x03', u'\x03\x04']
    >>> _code_tokens_to_unicode([[], []])
    [u'', u'']
    '''
    from collections import defaultdict
    prev_id = [0]

    def new_token():
        # argggh why didn't Python 2 figure out scoping
        # prev_id is only a list so we don't create a local variable shadowing it.
        prev_id[0] += 1
        if prev_id[0] >= 0x110000:
            import warnings
            warnings.warn('(probably while computing WER:) '
                          'Number of token types in collection exceeds maximum Unicode code point; '
                          'wrapping token ids. This may result in underestimating errors.')
            prev_id[0] = 1
        return unichr(prev_id[0])

    vocab = defaultdict(new_token)
    return [
        u''.join(vocab[t] for t in seq)
        for seq in seqs
    ]
