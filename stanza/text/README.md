# text

Package for manipulating text.

## text.vocab.Vocab

An abstraction for a vocabulary object that maps between word tokens and indices.

```ipython
In [10]: v = Vocab('UNK')

In [11]: v.add('the') # add a single word and return the index
Out[11]: 1

In [12]: v.word2index['the']
Out[12]: 1

In [13]: v.index2word[0]
Out[13]: 'UNK'

In [14]: v.words2indices(['the', 'cat', 'is', 'out'], add=True) # add multiple words and return the indices
Out[14]: [1, 2, 3, 4]

In [15]: v.words2indices('the cat is out of the bag'.split()) # convert words to indices
Out[15]: [1, 2, 3, 4, 0, 1, 0]

In [16]: v.indices2words([0, 1, 2, 3])
Out[16]: ['UNK', 'the', 'cat', 'is']
```