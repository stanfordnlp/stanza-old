# stanza

The Stanford NLP group's shared Python tools for deep learning.

This [Google Doc](https://docs.google.com/document/d/1tD0v8hNNilusNq632tYKNUn1g3Kfgu5dGNOtD9MA94Q/edit) logs what we are doing. Feel free to discuss ideas and review code via issues as well.


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