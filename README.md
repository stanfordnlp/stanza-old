# Stanza

Stanza is the Stanford NLP group’s shared repository for Python code. We currently plan to offer:

- **common objects used in NLP, e.g.**
    - a Vocabulary object mapping from strings to integers/vectors
- **tools for running experiments on the NLP cluster, e.g.**
    - a function for querying GPU device stats (to aid in selecting a GPU on the cluster)
    - a tool for plotting training curves from multiple jobs
    - a tool for interacting with an already running job via edits to a text file
- **an API for calling CoreNLP**

Stanza is still in early development. Interfaces and code organization will probably change substantially over the next few months. However, you can still benefit from useful code in Stanza right now by just copy-pasting parts you need.

- To request or discuss additional functionality, open a GitHub issue.
- To contribute code, make a pull request.

## For Stanford NLP members

Stanza is not meant to include every research project the group undertakes. If you have a standalone project that you would like to share with other people in the group, you can:

- request your own private repo under the [stanfordnlp GitHub account](https://github.com/stanfordnlp).
- share your code on [CodaLab](https://codalab.stanford.edu/).
- For targeted questions, ask on [Stanford NLP Overflow](http://nlp.stanford.edu/local/qa/) (use the `stanza` tag).

## Usage

You can install the package as follows:

```
git clone git@github.com:stanfordnlp/stanza.git
cd stanza
pip install -e .
```

To use the package, import it in your python code. An example would be:

```
from stanza.text.vocab import Vocab
v = Vocab()
```

## Development Guide

If you are adding a new module, please remember to add it to `setup.py`. You can create documentation for your
module by creating a README.md inside your subdirectory (it will also open by default on github when people browse
to your subdirectory).

Once you have developed and created unit tests for your module, please run

``` python
pip install -e .
```

to install your changes. Next, please make sure you have not regressed the code base:

```python
python setup.py test
```

## Contributors

- `Put your name here when you contribute to the repo!`
- Victor Zhong (vzhong)
- Kelvin Guu (kelvinguu)