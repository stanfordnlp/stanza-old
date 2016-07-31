Stanza
======

|Master Build Status| |Documentation Status|

Stanza is the Stanford NLP group’s shared repository for Python
infrastructure. The goal of Stanza is not to replace your modeling tools
of choice, but to offer implementations for common patterns useful for
machine learning experiments.

Usage
-----

You can install the package as follows:

::

    git clone git@github.com:stanfordnlp/stanza.git
    cd stanza
    pip install -e .

To use the package, import it in your python code. An example would be:

::

    from stanza.text.vocab import Vocab
    v = Vocab('UNK')

Please see the documentation for more use cases.

Documentation
-------------

Documentation is hosted on Read the Docs at
http://stanza.readthedocs.org/en/latest/. Stanza is still in early
development. Interfaces and code organization will probably change
substantially over the next few months.

Development Guide
-----------------

To request or discuss additional functionality, please open a GitHub
issue. We greatly appreciate pull requests!

Tests
~~~~~

Stanza has unit tests, doctests, and longer, integration tests. We ask that all
contributors run the unit tests and doctests before submitting pull requests:

.. code:: python

    python setup.py test

Doctests are the easiest way to write a test for new functionality, and serve
as helpful examples for how to use your code. See
`progress.py <stanza/research/progress.py>`__ for a simple example of a easily
testable module, or `summary.py <stanza/research/summary.py>`__ for a more
involved setup involving a mocked filesystem.

Adding a new module
~~~~~~~~~~~~~~~~~~~

If you are adding a new module, please remember to add it to
``setup.py`` as well as a corresponding ``.rst`` file in the ``docs``
directory.

Documentation
~~~~~~~~~~~~~

Documentation is generated via
`Sphinx <http://www.sphinx-doc.org/en/stable/>`__ using inline comments.
This means that the docstring in Python double both as interactive
documentation and standalone documentation. This also means that you
must format your docstring in RST. RST is very similar to Markdown.
There are many tutorials on the exact syntax, essentially you only need
to know the function parameter syntax which can be found
`here <http://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#auto-document-your-python-code>`__.
You can, of course, look at documentations for existing modules for
guidance as well. A good place to start is the ``text.dataset`` package.

To set up your environment such that you can generate docs locally:

::

    pip install sphinx sphinx-autobuild

If you introduced a new module, please auto-generate the docs:

::

    sphinx-apidoc -F -o docs stanza
    cd docs && make
    open _build/html/index.html

You most likely need to manually edit the `rst` file corresponding to your new module.

Our docs are `hosted on Readthedocs <https://readthedocs.org/projects/stanza/>`__. If you'd like admin access to the Readthedocs project, please contact Victor or Will.

Road Map
--------

-  common objects used in NLP

   -  [x] a Vocabulary object mapping from strings to integers/vectors

-  tools for running experiments on the NLP cluster

   -  [ ] a function for querying GPU device stats (to aid in selecting
      a GPU on the cluster)
   -  [ ] a tool for plotting training curves from multiple jobs
   -  [ ] a tool for interacting with an already running job via edits
      to a text file

-  [x] an API for calling CoreNLP

For Stanford NLP members
------------------------

Stanza is not meant to include every research project the group
undertakes. If you have a standalone project that you would like to
share with other people in the group, you can:

-  request your own private repo under the `stanfordnlp GitHub
   account <https://github.com/stanfordnlp>`__.
-  share your code on `CodaLab <https://codalab.stanford.edu/>`__.
-  For targeted questions, ask on `Stanford NLP
   Overflow <http://nlp.stanford.edu/local/qa/>`__ (use the ``stanza``
   tag).

Using `git subtree`
~~~~~~~~~~~~~~~~~~~

That said, it can be useful to add functionality to Stanza while you work in a
separate repo on a project that depends on Stanza. Since Stanza is under active
development, you will want to version-control the Stanza code that your code
uses. Probably the most effective way of accomplishing this is by using
``git subtree``.

``git subtree`` includes the source tree of another repo (in
this case, Stanza) as a directory within your repo (your cutting-edge
research), and keeps track of some metadata that allows you to keep that
directory in sync with the original Stanza code.  The main advantage of ``git
subtree`` is that you can modify the Stanza code locally, merge in updates, and
push your changes back to the Stanza repo to share them with the group. (``git
submodule`` doesn't allow this.)

It has some downsides to be aware of:

-  You have a copy of all of Stanza as part of your repo. For small projects,
   this could increase your repo size dramatically. (Note: you can keep the
   history of your repo from growing at the same rate as Stanza's by using
   squashed commits; it's only the size of the source tree that unavoidably
   bloats your project.)
-  Your repo's history will contain a merge commit every time you update Stanza
   from upstream. This can look ugly, especially in graphical viewers.

Still, ``subtree`` can be configured to be fairly easy to use, and the consensus
seems to be that it is superior to ``submodule`` (`<https://codingkilledthecat.wordpress.com/2012/04/28/why-your-company-shouldnt-use-git-submodules/>`__).

Here's one way to configure ``subtree`` so that you can include Stanza in
your repo and contribute your changes back to the master repo:

::

    # Add Stanza as a remote repo
    git remote add stanza http://<your github username>@github.com/stanfordnlp/stanza.git
    # Import the contents of the repo as a subtree
    git subtree add --prefix third-party/stanza stanza develop --squash
    # Put a symlink to the actual module somewhere where your code needs it
    ln -s third-party/stanza/stanza stanza
    # Add aliases for the two things you'll need to do with the subtree
    git config alias.stanza-update 'subtree pull --prefix third-party/stanza stanza develop --squash'
    git config alias.stanza-push 'subtree push --prefix third-party/stanza stanza develop'

After this, you can use the aliases to push and pull Stanza like so:

::

    git stanza-update
    git stanza-push

I [@futurulus] highly recommend a `topic branch/rebase workflow <https://randyfay.com/content/rebase-workflow-git>`__,
which will keep your history fairly clean besides those pesky subtree merge
commits:

::

    # Create a topic branch
    git checkout -b fix-stanza
    # <hack hack hack, make some commits>

    git checkout master
    # Update Stanza on master, should go smoothly because master doesn't
    # have any of your changes yet
    git stanza-update

    # Go back and replay your fixes on top of master changes
    git checkout fix-stanza
    git rebase master
    # You might need to resolve merge conflicts here

    # Add your rebased changes to master and push
    git checkout master
    git merge --ff-only fix-stanza
    git stanza-push
    # Done!
    git branch -d fix-stanza

.. |Master Build Status| image:: https://travis-ci.org/stanfordnlp/stanza.svg?branch=master
   :target: https://travis-ci.org/stanfordnlp/stanza
.. |Documentation Status| image:: https://readthedocs.org/projects/stanza/badge/?version=latest
   :target: http://stanza.readthedocs.org/en/latest/?badge=latest
