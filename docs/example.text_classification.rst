
Text Classification
===================

Author: Victor Zhong

We are going to tackle a relatively straight forward text classification
problem with Stanza and Tensorflow.

Dataset
-------

First, we'll grab the 20 newsgroup data, which is conviently downloaded
by ``sklearn``.

.. code:: python

    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')

Unsurprisingly, the 20 newsgroup data contains newgroup text from 20
topics. The topics are as follows:

.. code:: python

    classes = list(newsgroups_train.target_names)
    classes




.. parsed-literal::

    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']



We'll limit ourselves to two classes for sake of simplicity

.. code:: python

    classes = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=classes)
    from collections import Counter
    Counter([classes[t] for t in newsgroups_train.target])




.. parsed-literal::

    Counter({'alt.atheism': 480, 'soc.religion.christian': 599})



Here's an example from the dataset:

.. code:: python

    newsgroups_train.data[0]




.. parsed-literal::

    u'From: nigel.allen@canrem.com (Nigel Allen)\nSubject: library of congress to host dead sea scroll symposium april 21-22\nLines: 96\n\n\n Library of Congress to Host Dead Sea Scroll Symposium April 21-22\n To: National and Assignment desks, Daybook Editor\n Contact: John Sullivan, 202-707-9216, or Lucy Suddreth, 202-707-9191\n          both of the Library of Congress\n\n   WASHINGTON, April 19  -- A symposium on the Dead Sea \nScrolls will be held at the Library of Congress on Wednesday,\nApril 21, and Thursday, April 22.  The two-day program, cosponsored\nby the library and Baltimore Hebrew University, with additional\nsupport from the Project Judaica Foundation, will be held in the\nlibrary\'s Mumford Room, sixth floor, Madison Building.\n   Seating is limited, and admission to any session of the symposium\nmust be requested in writing (see Note A).\n   The symposium will be held one week before the public opening of a\nmajor exhibition, "Scrolls from the Dead Sea: The Ancient Library of\nQumran and Modern Scholarship," that opens at the Library of Congress\non April 29.  On view will be fragmentary scrolls and archaeological\nartifacts excavated at Qumran, on loan from the Israel Antiquities\nAuthority.  Approximately 50 items from Library of Congress special\ncollections will augment these materials.  The exhibition, on view in\nthe Madison Gallery, through Aug. 1, is made possible by a generous\ngift from the Project Judaica Foundation of Washington, D.C.\n   The Dead Sea Scrolls have been the focus of public and scholarly\ninterest since 1947, when they were discovered in the desert 13 miles\neast of Jerusalem.  The symposium will explore the origin and meaning\nof the scrolls and current scholarship.  Scholars from diverse\nacademic backgrounds and religious affiliations, will offer their\ndisparate views, ensuring a lively discussion.\n   The symposium schedule includes opening remarks on April 21, at\n2 p.m., by Librarian of Congress James H. Billington, and by\nDr. Norma Furst, president, Baltimore Hebrew University.  Co-chairing\nthe symposium are Joseph Baumgarten, professor of Rabbinic Literature\nand Institutions, Baltimore Hebrew University and Michael Grunberger,\nhead, Hebraic Section, Library of Congress.\n   Geza Vermes, professor emeritus of Jewish studies, Oxford\nUniversity, will give the keynote address on the current state of\nscroll research, focusing on where we stand today. On the second\nday, the closing address will be given by Shmaryahu Talmon, who will\npropose a research agenda, picking up the theme of how the Qumran\nstudies might proceed.\n   On Wednesday, April 21, other speakers will include:\n\n   -- Eugene Ulrich, professor of Hebrew Scriptures, University of\nNotre Dame and chief editor, Biblical Scrolls from Qumran, on "The\nBible at Qumran;"\n   -- Michael Stone, National Endowment for the Humanities\ndistinguished visiting professor of religious studies, University of\nRichmond, on "The Dead Sea Scrolls and the Pseudepigrapha."\n   -- From 5 p.m. to 6:30 p.m. a special preview of the exhibition\nwill be given to symposium participants and guests.\n\n   On Thursday, April 22, beginning at 9 a.m., speakers will include:\n\n   -- Magen Broshi, curator, shrine of the Book, Israel Museum,\nJerusalem, on "Qumran: The Archaeological Evidence;"\n   -- P. Kyle McCarter, Albright professor of Biblical and ancient\nnear Eastern studies, The Johns Hopkins University, on "The Copper\nScroll;"\n   -- Lawrence H. Schiffman, professor of Hebrew and Judaic studies,\nNew York University, on "The Dead Sea Scrolls and the History of\nJudaism;" and\n   -- James VanderKam, professor of theology, University of Notre\nDame, on "Messianism in the Scrolls and in Early Christianity."\n\n   The Thursday afternoon sessions, at 1:30 p.m., include:\n\n   -- Devorah Dimant, associate professor of Bible and Ancient Jewish\nThought, University of Haifa, on "Qumran Manuscripts: Library of a\nJewish Community;"\n   -- Norman Golb, Rosenberger professor of Jewish history and\ncivilization, Oriental Institute, University of Chicago, on "The\nCurrent Status of the Jerusalem Origin of the Scrolls;"\n   -- Shmaryahu Talmon, J.L. Magnas professor emeritus of Biblical\nstudies, Hebrew University, Jerusalem, on "The Essential \'Commune of\nthe Renewed Covenant\': How Should Qumran Studies Proceed?" will close\nthe symposium.\n\n   There will be ample time for question and answer periods at the\nend of each session.\n\n   Also on Wednesday, April 21, at 11 a.m.:\n   The Library of Congress and The Israel Antiquities Authority\nwill hold a lecture by Esther Boyd-Alkalay, consulting conservator,\nIsrael Antiquities Authority, on "Preserving the Dead Sea Scrolls"\nin the Mumford Room, LM-649, James Madison Memorial Building, The\nLibrary of Congress, 101 Independence Ave., S.E., Washington, D.C.\n    ------\n   NOTE A: For more information about admission to the symposium,\nplease contact, in writing, Dr. Michael Grunberger, head, Hebraic\nSection, African and Middle Eastern Division, Library of Congress,\nWashington, D.C. 20540.\n -30-\n--\nCanada Remote Systems - Toronto, Ontario\n416-629-7000/629-7044\n'



.. code:: python

    newsgroups_train.target[0]




.. parsed-literal::

    1



Notice that the target is already converted into a class index. Namely,
in this case the text belongs to the class:

.. code:: python

    classes[newsgroups_train.target[0]]




.. parsed-literal::

    'soc.religion.christian'



Annotating using CoreNLP
------------------------

If you do not have CoreNLP, download it from here:

http://stanfordnlp.github.io/CoreNLP/index.html#download

We are going to use the Java server feature of CoreNLP to annotate data
in python. In the CoreNLP directory, run the server:

.. code:: bash

    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

Next, we'll annotate an example to see how the server works.

.. code:: python

    from stanza.corenlp.client import Client
    
    client = Client()
    annotation = client.annotate(newsgroups_train.data[0], properties={'annotators': 'tokenize,ssplit,pos'})
    annotation['sentences'][0]




.. parsed-literal::

    {u'index': 0,
     u'parse': u'SENTENCE_SKIPPED_OR_UNPARSABLE',
     u'tokens': [{u'after': u'',
       u'before': u'',
       u'characterOffsetBegin': 0,
       u'characterOffsetEnd': 4,
       u'index': 1,
       u'originalText': u'From',
       u'pos': u'IN',
       u'word': u'From'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 4,
       u'characterOffsetEnd': 5,
       u'index': 2,
       u'originalText': u':',
       u'pos': u':',
       u'word': u':'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 6,
       u'characterOffsetEnd': 28,
       u'index': 3,
       u'originalText': u'nigel.allen@canrem.com',
       u'pos': u'NNP',
       u'word': u'nigel.allen@canrem.com'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 29,
       u'characterOffsetEnd': 30,
       u'index': 4,
       u'originalText': u'(',
       u'pos': u'-LRB-',
       u'word': u'-LRB-'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 30,
       u'characterOffsetEnd': 35,
       u'index': 5,
       u'originalText': u'Nigel',
       u'pos': u'NNP',
       u'word': u'Nigel'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 36,
       u'characterOffsetEnd': 41,
       u'index': 6,
       u'originalText': u'Allen',
       u'pos': u'NNP',
       u'word': u'Allen'},
      {u'after': u'\n',
       u'before': u'',
       u'characterOffsetBegin': 41,
       u'characterOffsetEnd': 42,
       u'index': 7,
       u'originalText': u')',
       u'pos': u'-RRB-',
       u'word': u'-RRB-'},
      {u'after': u'',
       u'before': u'\n',
       u'characterOffsetBegin': 43,
       u'characterOffsetEnd': 50,
       u'index': 8,
       u'originalText': u'Subject',
       u'pos': u'NNP',
       u'word': u'Subject'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 50,
       u'characterOffsetEnd': 51,
       u'index': 9,
       u'originalText': u':',
       u'pos': u':',
       u'word': u':'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 52,
       u'characterOffsetEnd': 59,
       u'index': 10,
       u'originalText': u'library',
       u'pos': u'NN',
       u'word': u'library'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 60,
       u'characterOffsetEnd': 62,
       u'index': 11,
       u'originalText': u'of',
       u'pos': u'IN',
       u'word': u'of'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 63,
       u'characterOffsetEnd': 71,
       u'index': 12,
       u'originalText': u'congress',
       u'pos': u'NN',
       u'word': u'congress'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 72,
       u'characterOffsetEnd': 74,
       u'index': 13,
       u'originalText': u'to',
       u'pos': u'TO',
       u'word': u'to'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 75,
       u'characterOffsetEnd': 79,
       u'index': 14,
       u'originalText': u'host',
       u'pos': u'NN',
       u'word': u'host'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 80,
       u'characterOffsetEnd': 84,
       u'index': 15,
       u'originalText': u'dead',
       u'pos': u'JJ',
       u'word': u'dead'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 85,
       u'characterOffsetEnd': 88,
       u'index': 16,
       u'originalText': u'sea',
       u'pos': u'NN',
       u'word': u'sea'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 89,
       u'characterOffsetEnd': 95,
       u'index': 17,
       u'originalText': u'scroll',
       u'pos': u'NN',
       u'word': u'scroll'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 96,
       u'characterOffsetEnd': 105,
       u'index': 18,
       u'originalText': u'symposium',
       u'pos': u'NN',
       u'word': u'symposium'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 106,
       u'characterOffsetEnd': 111,
       u'index': 19,
       u'originalText': u'april',
       u'pos': u'NNP',
       u'word': u'april'},
      {u'after': u'\n',
       u'before': u' ',
       u'characterOffsetBegin': 112,
       u'characterOffsetEnd': 117,
       u'index': 20,
       u'originalText': u'21-22',
       u'pos': u'CD',
       u'word': u'21-22'},
      {u'after': u'',
       u'before': u'\n',
       u'characterOffsetBegin': 118,
       u'characterOffsetEnd': 123,
       u'index': 21,
       u'originalText': u'Lines',
       u'pos': u'NNPS',
       u'word': u'Lines'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 123,
       u'characterOffsetEnd': 124,
       u'index': 22,
       u'originalText': u':',
       u'pos': u':',
       u'word': u':'},
      {u'after': u'\n\n\n ',
       u'before': u' ',
       u'characterOffsetBegin': 125,
       u'characterOffsetEnd': 127,
       u'index': 23,
       u'originalText': u'96',
       u'pos': u'CD',
       u'word': u'96'},
      {u'after': u' ',
       u'before': u'\n\n\n ',
       u'characterOffsetBegin': 131,
       u'characterOffsetEnd': 138,
       u'index': 24,
       u'originalText': u'Library',
       u'pos': u'NNP',
       u'word': u'Library'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 139,
       u'characterOffsetEnd': 141,
       u'index': 25,
       u'originalText': u'of',
       u'pos': u'IN',
       u'word': u'of'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 142,
       u'characterOffsetEnd': 150,
       u'index': 26,
       u'originalText': u'Congress',
       u'pos': u'NNP',
       u'word': u'Congress'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 151,
       u'characterOffsetEnd': 153,
       u'index': 27,
       u'originalText': u'to',
       u'pos': u'TO',
       u'word': u'to'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 154,
       u'characterOffsetEnd': 158,
       u'index': 28,
       u'originalText': u'Host',
       u'pos': u'NNP',
       u'word': u'Host'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 159,
       u'characterOffsetEnd': 163,
       u'index': 29,
       u'originalText': u'Dead',
       u'pos': u'NNP',
       u'word': u'Dead'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 164,
       u'characterOffsetEnd': 167,
       u'index': 30,
       u'originalText': u'Sea',
       u'pos': u'NNP',
       u'word': u'Sea'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 168,
       u'characterOffsetEnd': 174,
       u'index': 31,
       u'originalText': u'Scroll',
       u'pos': u'NNP',
       u'word': u'Scroll'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 175,
       u'characterOffsetEnd': 184,
       u'index': 32,
       u'originalText': u'Symposium',
       u'pos': u'NNP',
       u'word': u'Symposium'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 185,
       u'characterOffsetEnd': 190,
       u'index': 33,
       u'originalText': u'April',
       u'pos': u'NNP',
       u'word': u'April'},
      {u'after': u'\n ',
       u'before': u' ',
       u'characterOffsetBegin': 191,
       u'characterOffsetEnd': 196,
       u'index': 34,
       u'originalText': u'21-22',
       u'pos': u'CD',
       u'word': u'21-22'},
      {u'after': u'',
       u'before': u'\n ',
       u'characterOffsetBegin': 198,
       u'characterOffsetEnd': 200,
       u'index': 35,
       u'originalText': u'To',
       u'pos': u'TO',
       u'word': u'To'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 200,
       u'characterOffsetEnd': 201,
       u'index': 36,
       u'originalText': u':',
       u'pos': u':',
       u'word': u':'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 202,
       u'characterOffsetEnd': 210,
       u'index': 37,
       u'originalText': u'National',
       u'pos': u'NNP',
       u'word': u'National'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 211,
       u'characterOffsetEnd': 214,
       u'index': 38,
       u'originalText': u'and',
       u'pos': u'CC',
       u'word': u'and'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 215,
       u'characterOffsetEnd': 225,
       u'index': 39,
       u'originalText': u'Assignment',
       u'pos': u'NNP',
       u'word': u'Assignment'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 226,
       u'characterOffsetEnd': 231,
       u'index': 40,
       u'originalText': u'desks',
       u'pos': u'NNS',
       u'word': u'desks'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 231,
       u'characterOffsetEnd': 232,
       u'index': 41,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 233,
       u'characterOffsetEnd': 240,
       u'index': 42,
       u'originalText': u'Daybook',
       u'pos': u'NNP',
       u'word': u'Daybook'},
      {u'after': u'\n ',
       u'before': u' ',
       u'characterOffsetBegin': 241,
       u'characterOffsetEnd': 247,
       u'index': 43,
       u'originalText': u'Editor',
       u'pos': u'NNP',
       u'word': u'Editor'},
      {u'after': u'',
       u'before': u'\n ',
       u'characterOffsetBegin': 249,
       u'characterOffsetEnd': 256,
       u'index': 44,
       u'originalText': u'Contact',
       u'pos': u'NN',
       u'word': u'Contact'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 256,
       u'characterOffsetEnd': 257,
       u'index': 45,
       u'originalText': u':',
       u'pos': u':',
       u'word': u':'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 258,
       u'characterOffsetEnd': 262,
       u'index': 46,
       u'originalText': u'John',
       u'pos': u'NNP',
       u'word': u'John'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 263,
       u'characterOffsetEnd': 271,
       u'index': 47,
       u'originalText': u'Sullivan',
       u'pos': u'NNP',
       u'word': u'Sullivan'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 271,
       u'characterOffsetEnd': 272,
       u'index': 48,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 273,
       u'characterOffsetEnd': 285,
       u'index': 49,
       u'originalText': u'202-707-9216',
       u'pos': u'CD',
       u'word': u'202-707-9216'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 285,
       u'characterOffsetEnd': 286,
       u'index': 50,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 287,
       u'characterOffsetEnd': 289,
       u'index': 51,
       u'originalText': u'or',
       u'pos': u'CC',
       u'word': u'or'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 290,
       u'characterOffsetEnd': 294,
       u'index': 52,
       u'originalText': u'Lucy',
       u'pos': u'NNP',
       u'word': u'Lucy'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 295,
       u'characterOffsetEnd': 303,
       u'index': 53,
       u'originalText': u'Suddreth',
       u'pos': u'NNP',
       u'word': u'Suddreth'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 303,
       u'characterOffsetEnd': 304,
       u'index': 54,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u'\n          ',
       u'before': u' ',
       u'characterOffsetBegin': 305,
       u'characterOffsetEnd': 317,
       u'index': 55,
       u'originalText': u'202-707-9191',
       u'pos': u'CD',
       u'word': u'202-707-9191'},
      {u'after': u' ',
       u'before': u'\n          ',
       u'characterOffsetBegin': 328,
       u'characterOffsetEnd': 332,
       u'index': 56,
       u'originalText': u'both',
       u'pos': u'DT',
       u'word': u'both'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 333,
       u'characterOffsetEnd': 335,
       u'index': 57,
       u'originalText': u'of',
       u'pos': u'IN',
       u'word': u'of'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 336,
       u'characterOffsetEnd': 339,
       u'index': 58,
       u'originalText': u'the',
       u'pos': u'DT',
       u'word': u'the'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 340,
       u'characterOffsetEnd': 347,
       u'index': 59,
       u'originalText': u'Library',
       u'pos': u'NNP',
       u'word': u'Library'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 348,
       u'characterOffsetEnd': 350,
       u'index': 60,
       u'originalText': u'of',
       u'pos': u'IN',
       u'word': u'of'},
      {u'after': u'\n\n   ',
       u'before': u' ',
       u'characterOffsetBegin': 351,
       u'characterOffsetEnd': 359,
       u'index': 61,
       u'originalText': u'Congress',
       u'pos': u'NNP',
       u'word': u'Congress'},
      {u'after': u'',
       u'before': u'\n\n   ',
       u'characterOffsetBegin': 364,
       u'characterOffsetEnd': 374,
       u'index': 62,
       u'originalText': u'WASHINGTON',
       u'pos': u'NNP',
       u'word': u'WASHINGTON'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 374,
       u'characterOffsetEnd': 375,
       u'index': 63,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 376,
       u'characterOffsetEnd': 381,
       u'index': 64,
       u'originalText': u'April',
       u'pos': u'NNP',
       u'word': u'April'},
      {u'after': u'  ',
       u'before': u' ',
       u'characterOffsetBegin': 382,
       u'characterOffsetEnd': 384,
       u'index': 65,
       u'originalText': u'19',
       u'pos': u'CD',
       u'word': u'19'},
      {u'after': u' ',
       u'before': u'  ',
       u'characterOffsetBegin': 386,
       u'characterOffsetEnd': 388,
       u'index': 66,
       u'originalText': u'--',
       u'pos': u':',
       u'word': u'--'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 389,
       u'characterOffsetEnd': 390,
       u'index': 67,
       u'originalText': u'A',
       u'pos': u'DT',
       u'word': u'A'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 391,
       u'characterOffsetEnd': 400,
       u'index': 68,
       u'originalText': u'symposium',
       u'pos': u'NN',
       u'word': u'symposium'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 401,
       u'characterOffsetEnd': 403,
       u'index': 69,
       u'originalText': u'on',
       u'pos': u'IN',
       u'word': u'on'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 404,
       u'characterOffsetEnd': 407,
       u'index': 70,
       u'originalText': u'the',
       u'pos': u'DT',
       u'word': u'the'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 408,
       u'characterOffsetEnd': 412,
       u'index': 71,
       u'originalText': u'Dead',
       u'pos': u'NNP',
       u'word': u'Dead'},
      {u'after': u' \n',
       u'before': u' ',
       u'characterOffsetBegin': 413,
       u'characterOffsetEnd': 416,
       u'index': 72,
       u'originalText': u'Sea',
       u'pos': u'NNP',
       u'word': u'Sea'},
      {u'after': u' ',
       u'before': u' \n',
       u'characterOffsetBegin': 418,
       u'characterOffsetEnd': 425,
       u'index': 73,
       u'originalText': u'Scrolls',
       u'pos': u'NNP',
       u'word': u'Scrolls'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 426,
       u'characterOffsetEnd': 430,
       u'index': 74,
       u'originalText': u'will',
       u'pos': u'MD',
       u'word': u'will'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 431,
       u'characterOffsetEnd': 433,
       u'index': 75,
       u'originalText': u'be',
       u'pos': u'VB',
       u'word': u'be'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 434,
       u'characterOffsetEnd': 438,
       u'index': 76,
       u'originalText': u'held',
       u'pos': u'VBN',
       u'word': u'held'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 439,
       u'characterOffsetEnd': 441,
       u'index': 77,
       u'originalText': u'at',
       u'pos': u'IN',
       u'word': u'at'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 442,
       u'characterOffsetEnd': 445,
       u'index': 78,
       u'originalText': u'the',
       u'pos': u'DT',
       u'word': u'the'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 446,
       u'characterOffsetEnd': 453,
       u'index': 79,
       u'originalText': u'Library',
       u'pos': u'NNP',
       u'word': u'Library'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 454,
       u'characterOffsetEnd': 456,
       u'index': 80,
       u'originalText': u'of',
       u'pos': u'IN',
       u'word': u'of'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 457,
       u'characterOffsetEnd': 465,
       u'index': 81,
       u'originalText': u'Congress',
       u'pos': u'NNP',
       u'word': u'Congress'},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 466,
       u'characterOffsetEnd': 468,
       u'index': 82,
       u'originalText': u'on',
       u'pos': u'IN',
       u'word': u'on'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 469,
       u'characterOffsetEnd': 478,
       u'index': 83,
       u'originalText': u'Wednesday',
       u'pos': u'NNP',
       u'word': u'Wednesday'},
      {u'after': u'\n',
       u'before': u'',
       u'characterOffsetBegin': 478,
       u'characterOffsetEnd': 479,
       u'index': 84,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u' ',
       u'before': u'\n',
       u'characterOffsetBegin': 480,
       u'characterOffsetEnd': 485,
       u'index': 85,
       u'originalText': u'April',
       u'pos': u'NNP',
       u'word': u'April'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 486,
       u'characterOffsetEnd': 488,
       u'index': 86,
       u'originalText': u'21',
       u'pos': u'CD',
       u'word': u'21'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 488,
       u'characterOffsetEnd': 489,
       u'index': 87,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 490,
       u'characterOffsetEnd': 493,
       u'index': 88,
       u'originalText': u'and',
       u'pos': u'CC',
       u'word': u'and'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 494,
       u'characterOffsetEnd': 502,
       u'index': 89,
       u'originalText': u'Thursday',
       u'pos': u'NNP',
       u'word': u'Thursday'},
      {u'after': u' ',
       u'before': u'',
       u'characterOffsetBegin': 502,
       u'characterOffsetEnd': 503,
       u'index': 90,
       u'originalText': u',',
       u'pos': u',',
       u'word': u','},
      {u'after': u' ',
       u'before': u' ',
       u'characterOffsetBegin': 504,
       u'characterOffsetEnd': 509,
       u'index': 91,
       u'originalText': u'April',
       u'pos': u'NNP',
       u'word': u'April'},
      {u'after': u'',
       u'before': u' ',
       u'characterOffsetBegin': 510,
       u'characterOffsetEnd': 512,
       u'index': 92,
       u'originalText': u'22',
       u'pos': u'CD',
       u'word': u'22'},
      {u'after': u'  ',
       u'before': u'',
       u'characterOffsetBegin': 512,
       u'characterOffsetEnd': 513,
       u'index': 93,
       u'originalText': u'.',
       u'pos': u'.',
       u'word': u'.'}]}



That was rather long, but the gist is that the annotation is organized
into sentences, which is then organized into tokens. Each token carries
a number of annotations (we've only asked for the POS tags).

.. code:: python

    for token in annotation['sentences'][0]['tokens']:
        print token['word'], token['pos']


.. parsed-literal::

    From IN
    : :
    nigel.allen@canrem.com NNP
    -LRB- -LRB-
    Nigel NNP
    Allen NNP
    -RRB- -RRB-
    Subject NNP
    : :
    library NN
    of IN
    congress NN
    to TO
    host NN
    dead JJ
    sea NN
    scroll NN
    symposium NN
    april NNP
    21-22 CD
    Lines NNPS
    : :
    96 CD
    Library NNP
    of IN
    Congress NNP
    to TO
    Host NNP
    Dead NNP
    Sea NNP
    Scroll NNP
    Symposium NNP
    April NNP
    21-22 CD
    To TO
    : :
    National NNP
    and CC
    Assignment NNP
    desks NNS
    , ,
    Daybook NNP
    Editor NNP
    Contact NN
    : :
    John NNP
    Sullivan NNP
    , ,
    202-707-9216 CD
    , ,
    or CC
    Lucy NNP
    Suddreth NNP
    , ,
    202-707-9191 CD
    both DT
    of IN
    the DT
    Library NNP
    of IN
    Congress NNP
    WASHINGTON NNP
    , ,
    April NNP
    19 CD
    -- :
    A DT
    symposium NN
    on IN
    the DT
    Dead NNP
    Sea NNP
    Scrolls NNP
    will MD
    be VB
    held VBN
    at IN
    the DT
    Library NNP
    of IN
    Congress NNP
    on IN
    Wednesday NNP
    , ,
    April NNP
    21 CD
    , ,
    and CC
    Thursday NNP
    , ,
    April NNP
    22 CD
    . .


For our purpose, we're actually going to just take the document as a
long sequence of words as opposed to a sequence of sequences (eg. a list
of sentences of words). We'll do this by passing in the
``ssplit.isOneSentence`` flag.

.. code:: python

    docs = []
    labels = []
    for doc, label in zip(newsgroups_train.data, newsgroups_train.target):
        try:
            annotation = client.annotate(doc, properties={'annotators': 'tokenize,ssplit', 'ssplit.isOneSentence': True})
            docs.append([t['word'] for t in annotation['sentences'][0]['tokens']])
            labels.append(label)
        except Exception as e:
            pass  # we're going to punt and ignore unicode errors...
    print len(docs), len(labels)


.. parsed-literal::

    1074 1074


We'll create a lightweight dataset object out of this. A ``Dataset`` is
really a glorified dictionary of fields, where each field corresponds to
an attribute of the examples in the dataset.

.. code:: python

    from stanza.text.dataset import Dataset
    from pprint import pprint
    dataset = Dataset({'X': docs, 'Y': labels})
    
    # dataset supports, amongst other functionalities, shuffling:
    print dataset.shuffle()
    
    # indexing of a single element
    pprint(dataset[0])
    
    # indexing of multiple elements
    pprint(dataset[:2])
    
    n_train = int(0.7 * len(dataset))
    train = Dataset(dataset[:n_train])
    test = Dataset(dataset[n_train:])
    
    print 'train: {}, test: {}'.format(len(train), len(test))


.. parsed-literal::

    Dataset(Y, X)
    OrderedDict([('Y', 1), ('X', [u'From', u':', u'seanna@bnr.ca', u'-LRB-', u'Seanna', u'-LRB-', u'S.M.', u'-RRB-', u'Watson', u'-RRB-', u'Subject', u':', u'Re', u':', u'``', u'Accepting', u'Jeesus', u'in', u'your', u'heart', u'...', u"''", u'Organization', u':', u'Bell-Northern', u'Research', u',', u'Ottawa', u',', u'Canada', u'Lines', u':', u'38', u'-LCB-', u'Dan', u'Johnson', u'asked', u'for', u'evidence', u'that', u'the', u'most', u'effective', u'abuse', u'recovery', u'programs', u'involve', u'meeting', u'people', u"'s", u'spiritual', u'needs', u'.', u'I', u'responded', u':', u'In', u'12-step', u'programs', u'-LRB-', u'like', u'Alcoholics', u'Anonymous', u'-RRB-', u',', u'one', u'of', u'the', u'steps', u'involves', u'acknowleding', u'a', u'``', u'higher', u'power', u"''", u'.', u'AA', u'and', u'other', u'12-step', u'abuse', u'-', u'recovery', u'programs', u'are', u'acknowledged', u'as', u'being', u'among', u'the', u'most', u'effective', u'.', u'-RCB-', u'Dan', u'Johnson', u'clarified', u':', u'>', u'What', u'I', u'was', u'asking', u'is', u'this', u':', u'>', u'>', u'Please', u'show', u'me', u'that', u'the', u'most', u'effective', u'substance-absure', u'recovery', u'>', u'programs', u'involve', u'meetinsg', u'peoples', u"'", u'spiritual', u'needs', u',', u'rather', u'than', u'>', u'merely', u'attempting', u'to', u'fill', u'peoples', u"'", u'spiritual', u'needs', u'as', u'percieved', u'>', u'by', u'the', u'people', u',', u'A.A', u',', u'S.R.C.', u'regulars', u',', u'or', u'snoopy', u'.', u'You', u'are', u'asking', u'me', u'to', u'provide', u'objective', u'proof', u'for', u'the', u'existence', u'of', u'God', u'.', u'I', u'never', u'claimed', u'to', u'be', u'able', u'to', u'do', u'this', u';', u'in', u'fact', u'I', u'do', u'not', u'believe', u'that', u'it', u'is', u'possible', u'to', u'do', u'so', u'.', u'I', u'consider', u'the', u'existence', u'of', u'God', u'to', u'be', u'a', u'premise', u'or', u'assumption', u'that', u'underlies', u'my', u'philosophy', u'of', u'life', u'.', u'It', u'comes', u'down', u'to', u'a', u'matter', u'of', u'faith', u'.', u'If', u'I', u'were', u"n't", u'a', u'Christian', u',', u'I', u'would', u'be', u'an', u'agnostic', u',', u'but', u'I', u'have', u'sufficient', u'subjective', u'evidence', u'to', u'justify', u'and', u'sustain', u'my', u'relationship', u'with', u'God', u'.', u'Again', u'this', u'is', u'a', u'matter', u'of', u'premises', u'and', u'assumptions', u'.', u'I', u'assume', u'that', u'there', u'is', u'more', u'to', u'``', u'life', u',', u'the', u'universe', u'and', u'everything', u"''", u'than', u'materialism', u';', u'ie', u'that', u'spirituality', u'exists', u'.', u'This', u'assumption', u'answers', u'the', u'question', u'about', u'why', u'I', u'have', u'apparent', u'spiritual', u'needs', u'.', u'I', u'find', u'this', u'assumption', u'consistent', u'with', u'my', u'subsequent', u'observat', u'-', u'ions', u'.', u'I', u'then', u'find', u'that', u'God', u'fills', u'these', u'spiritual', u'needs', u'.', u'But', u'I', u'can', u'not', u'objectively', u'prove', u'the', u'difference', u'between', u'apparent', u'filling', u'of', u'imagined', u'spiritual', u'needs', u'and', u'real', u'filling', u'of', u'real', u'spiritual', u'needs', u'.', u'Nor', u'can', u'I', u'prove', u'to', u'another', u'person', u'that', u'_', u'they', u'_', u'have', u'spiritual', u'needs', u'.', u'==', u'Seanna', u'Watson', u'Bell-Northern', u'Research', u',', u'|', u'Pray', u'that', u'at', u'the', u'end', u'of', u'living', u',', u'-LRB-', u'seanna@bnr.ca', u'-RRB-', u'Ottawa', u',', u'Ontario', u',', u'Canada', u'|', u'Of', u'philosophies', u'and', u'creeds', u',', u'|', u'God', u'will', u'find', u'his', u'people', u'busy', u'Opinion', u',', u'what', u'opinions', u'?', u'Oh', u'*', u'these', u'*', u'opinions', u'.', u'|', u'Planting', u'trees', u'and', u'sowing', u'seeds', u'.', u'No', u',', u'they', u"'re", u'not', u'BNR', u"'s", u',', u'they', u"'re", u'mine', u'.', u'|', u'I', u'knew', u'I', u"'d", u'left', u'them', u'somewhere', u'.', u'|', u'--', u'Fred', u'Kaan'])])
    OrderedDict([('Y', [1, 0]), ('X', [[u'From', u':', u'seanna@bnr.ca', u'-LRB-', u'Seanna', u'-LRB-', u'S.M.', u'-RRB-', u'Watson', u'-RRB-', u'Subject', u':', u'Re', u':', u'``', u'Accepting', u'Jeesus', u'in', u'your', u'heart', u'...', u"''", u'Organization', u':', u'Bell-Northern', u'Research', u',', u'Ottawa', u',', u'Canada', u'Lines', u':', u'38', u'-LCB-', u'Dan', u'Johnson', u'asked', u'for', u'evidence', u'that', u'the', u'most', u'effective', u'abuse', u'recovery', u'programs', u'involve', u'meeting', u'people', u"'s", u'spiritual', u'needs', u'.', u'I', u'responded', u':', u'In', u'12-step', u'programs', u'-LRB-', u'like', u'Alcoholics', u'Anonymous', u'-RRB-', u',', u'one', u'of', u'the', u'steps', u'involves', u'acknowleding', u'a', u'``', u'higher', u'power', u"''", u'.', u'AA', u'and', u'other', u'12-step', u'abuse', u'-', u'recovery', u'programs', u'are', u'acknowledged', u'as', u'being', u'among', u'the', u'most', u'effective', u'.', u'-RCB-', u'Dan', u'Johnson', u'clarified', u':', u'>', u'What', u'I', u'was', u'asking', u'is', u'this', u':', u'>', u'>', u'Please', u'show', u'me', u'that', u'the', u'most', u'effective', u'substance-absure', u'recovery', u'>', u'programs', u'involve', u'meetinsg', u'peoples', u"'", u'spiritual', u'needs', u',', u'rather', u'than', u'>', u'merely', u'attempting', u'to', u'fill', u'peoples', u"'", u'spiritual', u'needs', u'as', u'percieved', u'>', u'by', u'the', u'people', u',', u'A.A', u',', u'S.R.C.', u'regulars', u',', u'or', u'snoopy', u'.', u'You', u'are', u'asking', u'me', u'to', u'provide', u'objective', u'proof', u'for', u'the', u'existence', u'of', u'God', u'.', u'I', u'never', u'claimed', u'to', u'be', u'able', u'to', u'do', u'this', u';', u'in', u'fact', u'I', u'do', u'not', u'believe', u'that', u'it', u'is', u'possible', u'to', u'do', u'so', u'.', u'I', u'consider', u'the', u'existence', u'of', u'God', u'to', u'be', u'a', u'premise', u'or', u'assumption', u'that', u'underlies', u'my', u'philosophy', u'of', u'life', u'.', u'It', u'comes', u'down', u'to', u'a', u'matter', u'of', u'faith', u'.', u'If', u'I', u'were', u"n't", u'a', u'Christian', u',', u'I', u'would', u'be', u'an', u'agnostic', u',', u'but', u'I', u'have', u'sufficient', u'subjective', u'evidence', u'to', u'justify', u'and', u'sustain', u'my', u'relationship', u'with', u'God', u'.', u'Again', u'this', u'is', u'a', u'matter', u'of', u'premises', u'and', u'assumptions', u'.', u'I', u'assume', u'that', u'there', u'is', u'more', u'to', u'``', u'life', u',', u'the', u'universe', u'and', u'everything', u"''", u'than', u'materialism', u';', u'ie', u'that', u'spirituality', u'exists', u'.', u'This', u'assumption', u'answers', u'the', u'question', u'about', u'why', u'I', u'have', u'apparent', u'spiritual', u'needs', u'.', u'I', u'find', u'this', u'assumption', u'consistent', u'with', u'my', u'subsequent', u'observat', u'-', u'ions', u'.', u'I', u'then', u'find', u'that', u'God', u'fills', u'these', u'spiritual', u'needs', u'.', u'But', u'I', u'can', u'not', u'objectively', u'prove', u'the', u'difference', u'between', u'apparent', u'filling', u'of', u'imagined', u'spiritual', u'needs', u'and', u'real', u'filling', u'of', u'real', u'spiritual', u'needs', u'.', u'Nor', u'can', u'I', u'prove', u'to', u'another', u'person', u'that', u'_', u'they', u'_', u'have', u'spiritual', u'needs', u'.', u'==', u'Seanna', u'Watson', u'Bell-Northern', u'Research', u',', u'|', u'Pray', u'that', u'at', u'the', u'end', u'of', u'living', u',', u'-LRB-', u'seanna@bnr.ca', u'-RRB-', u'Ottawa', u',', u'Ontario', u',', u'Canada', u'|', u'Of', u'philosophies', u'and', u'creeds', u',', u'|', u'God', u'will', u'find', u'his', u'people', u'busy', u'Opinion', u',', u'what', u'opinions', u'?', u'Oh', u'*', u'these', u'*', u'opinions', u'.', u'|', u'Planting', u'trees', u'and', u'sowing', u'seeds', u'.', u'No', u',', u'they', u"'re", u'not', u'BNR', u"'s", u',', u'they', u"'re", u'mine', u'.', u'|', u'I', u'knew', u'I', u"'d", u'left', u'them', u'somewhere', u'.', u'|', u'--', u'Fred', u'Kaan'], [u'From', u':', u'chrisb@seachg.com', u'-LRB-', u'Chris', u'Blask', u'-RRB-', u'Subject', u':', u'Re', u':', u'islamic', u'authority', u'over', u'women', u'Reply-To', u':', u'chrisb@seachg.com', u'-LRB-', u'Chris', u'Blask', u'-RRB-', u'Organization', u':', u'Me', u',', u'Mississauga', u',', u'Ontario', u',', u'Canada', u'Lines', u':', u'78', u'snm6394@ultb.isc.rit.edu', u'-LRB-', u'S.N.', u'Mozumder', u'-RRB-', u'writes', u':', u'>', u'In', u'article', u'<1993Apr7.163445.1203@wam.umd.edu>', u'west@next02.wam.umd.edu', u'writes', u':', u'>>', u'>', u'>>', u'And', u'belief', u'causes', u'far', u'more', u'horrors', u'.', u'>>', u'>', u'>>', u'Crusades', u',', u'>>', u'>', u'>>', u'the', u'emasculation', u'and', u'internment', u'of', u'Native', u'Americans', u',', u'>>', u'>', u'>>', u'the', u'killing', u'of', u'various', u'tribes', u'in', u'South', u'America', u'.', u'>>', u'>', u'>', u'-', u'the', u'Inquisition', u'>>', u'>', u'>', u'-', u'the', u'Counter-reformation', u'and', u'the', u'wars', u'that', u'followed', u'>>', u'>', u'>', u'-', u'the', u'Salem', u'witch', u'trials', u'>>', u'>', u'>', u'-', u'the', u'European', u'witch', u'hunts', u'>>', u'>', u'>', u'-', u'the', u'holy', u'wars', u'of', u'the', u'middle', u'east', u'>>', u'>', u'>', u'-', u'the', u'colonization/destruction', u'of', u'Africa', u'>>', u'>', u'>', u'-', u'the', u'wars', u'between', u'Christianity', u'and', u'Islam', u'-LRB-', u'post', u'crusade', u'-RRB-', u'>>', u'>', u'>', u'-', u'the', u'genocide', u'-LRB-', u'biblical', u'-RRB-', u'of', u'the', u'Canaanites', u'and', u'Philistines', u'>>', u'>', u'>', u'-', u'Aryian', u'invasion', u'of', u'India', u'>>', u'>', u'>', u'-', u'the', u'attempted', u'genocide', u'of', u'Jews', u'by', u'Nazi', u'Germany', u'>>', u'>', u'>', u'-', u'the', u'current', u'missionary', u'assaults', u'on', u'tribes', u'in', u'Africa', u'>>', u'>', u'>>', u'>', u'I', u'think', u'all', u'the', u'horrors', u'you', u'mentioned', u'are', u'due', u'to', u'*', u'lack', u'*', u'of', u'people', u'>>', u'>', u'following', u'religion', u'.', u'.', u'd.', u'>', u'By', u'lack', u'of', u'people', u'following', u'religion', u'I', u'also', u'include', u'fanatics', u'-', u'people', u'>', u'that', u'do', u"n't", u'know', u'what', u'they', u'are', u'following', u'.', u'.', u'd.', u'>', u'So', u'how', u'do', u'you', u'know', u'that', u'you', u'were', u'right', u'?', u'>', u'Why', u'are', u'you', u'trying', u'to', u'shove', u'down', u'my', u'throat', u'that', u'religion', u'causes', u'horrors', u'.', u'>', u'It', u'really', u'covers', u'yourself', u'-', u'something', u'false', u'to', u'save', u'yourself', u'.', u'>', u'>', u'Peace', u',', u'>', u'>', u'Bobby', u'Mozumder', u'>', u'I', u'just', u'thought', u'of', u'another', u'one', u',', u'in', u'the', u'Bible', u',', u'so', u'it', u"'s", u'definately', u'not', u'because', u'of', u'*', u'lack', u'*', u'of', u'religion', u'.', u'The', u'Book', u'of', u'Esther', u'-LRB-', u'which', u'I', u'read', u'the', u'other', u'day', u'for', u'other', u'reasons', u'-RRB-', u'describes', u'the', u'origin', u'of', u'Pur', u'`', u'im', u',', u'a', u'Jewish', u'celbration', u'of', u'joy', u'and', u'peace', u'.', u'The', u'long', u'and', u'short', u'of', u'the', u'story', u'is', u'that', u'75,000', u'people', u'were', u'killed', u'when', u'people', u'were', u'tripping', u'over', u'all', u'of', u'the', u'peacefull', u'solutions', u'lying', u'about', u'-LRB-', u'you', u'could', u"n't", u'swing', u'a', u'sacred', u'cow', u'without', u'slammin', u'into', u'a', u'nice', u',', u'peaceful', u'solution', u'.', u'-RRB-', u'`', u'Course', u'Joshua', u'and', u'the', u'jawbone', u'of', u'an', u'ass', u'spring', u'to', u'mind', u'...', u'I', u'agree', u'with', u'Bobby', u'this', u'far', u':', u'religion', u'as', u'it', u'is', u'used', u'to', u'kill', u'large', u'numbers', u'of', u'people', u'is', u'usually', u'not', u'used', u'in', u'the', u'form', u'or', u'manner', u'that', u'it', u'was', u'originally', u'intended', u'for', u'.', u'That', u'does', u"n't", u'reduce', u'the', u'number', u'of', u'deaths', u'directly', u'caused', u'by', u'religion', u',', u'it', u'is', u'just', u'a', u'minor', u'observation', u'of', u'the', u'fact', u'that', u'there', u'is', u'almost', u'nothing', u'pure', u'in', u'the', u'Universe', u'.', u'The', u'very', u'act', u'of', u'honestly', u'attempting', u'to', u'find', u'true', u'meaning', u'in', u'religious', u'teaching', u'has', u'many', u'times', u'inspired', u'hatred', u'and', u'led', u'to', u'war', u'.', u'Many', u'people', u'have', u'been', u'led', u'by', u'religious', u'leaders', u'more', u'involved', u'in', u'their', u'own', u'stomache-contentsthan', u'in', u'any', u'absolute', u'truth', u',', u'and', u'have', u'therefore', u'been', u'driven', u'to', u'kill', u'by', u'their', u'leaders', u'.', u'The', u'point', u'is', u'that', u'there', u'are', u'many', u'things', u'involved', u'in', u'religion', u'that', u'often', u'lead', u'to', u'war', u'.', u'Whether', u'these', u'things', u'are', u'a', u'part', u'of', u'religion', u',', u'an', u'unpleasant', u'side', u'effect', u'or', u'-LRB-', u'as', u'Bobby', u'would', u'have', u'it', u'-RRB-', u'the', u'result', u'of', u'people', u'switching', u'between', u'Religion', u'and', u'Atheism', u'spontaneously', u',', u'the', u'results', u'are', u'the', u'same', u'.', u'@Religious', u'groups', u'have', u'long', u'been', u'involved', u'in', u'the', u'majority', u'of', u'the', u'bloodiest', u'parts', u'of', u'Man', u"'s", u'history', u'.', u'@', u'Atheists', u',', u'on', u'the', u'other', u'hand', u'-LRB-', u'preen', u',', u'preen', u'-RRB-', u'are', u'typically', u'not', u'an', u'ideological', u'social', u'caste', u',', u'nor', u'are', u'they', u'driven', u'to', u'organize', u'and', u'spread', u'their', u'beliefs', u'.', u'The', u'overuse', u'of', u'Nazism', u'and', u'Stalinism', u'just', u'show', u'how', u'true', u'this', u'is', u':', u'Two', u'groups', u'with', u'very', u'clear', u'and', u'specific', u'ideologies', u'using', u'religious', u'persecution', u'to', u'further', u'their', u'means', u'.', u'Anyone', u'who', u'can', u'not', u'see', u'the', u'obvious', u'-', u'namely', u'that', u'these', u'were', u'groups', u'founded', u'for', u'reasons', u'*', u'entirely', u'*', u'their', u'own', u',', u'who', u'used', u'religious', u'persecution', u'not', u'because', u'of', u'any', u'belief', u'system', u'but', u'because', u'it', u'made', u'them', u'more', u'powerfull', u'-', u'is', u'trying', u'too', u'hard', u'.', u'Basically', u',', u'Bobby', u'uses', u'these', u'examples', u'because', u'there', u'are', u'so', u'few', u'wars', u'that', u'were', u'*', u'not', u'*', u'*', u'specifically', u'*', u'fought', u'over', u'religion', u'that', u'he', u'does', u'not', u'have', u'many', u'choices', u'.', u'Well', u',', u'I', u"'m", u'off', u'to', u'Key', u'West', u'where', u'the', u'only', u'flames', u'are', u'heating', u'the', u'bottom', u'of', u'little', u'silver', u'butter-dishes', u'.', u'-', u'ciao', u'-', u'chris', u'blask']])])
    train: 751, test: 323


Creating vocabulary and mapping to vector space
-----------------------------------------------

Stanza provides means to convert words to vocabularies (eg. map to
indices and back). We also provide convienient means of loading
pretrained embeddings such as ``Senna`` and ``Glove``.

.. code:: python

    from stanza.text.vocab import Vocab
    vocab = Vocab('***UNK***')
    vocab




.. parsed-literal::

    OrderedDict([('***UNK***', 0)])



We'll try our hands at some conversions:

.. code:: python

    sents = ['I like cats and dogs', 'I like nothing', 'I like cats and nothing else']
    inds = []
    for s in sents[:2]:
        inds.append(vocab.update(s.split()))
    inds.append(vocab.words2indices(sents[2].split()))
    
    for s, ind in zip(sents, inds):
        print 'read {}, which got mapped to indices {}\nrecovered:{}'.format(s, ind, vocab.indices2words(ind))


.. parsed-literal::

    read I like cats and dogs, which got mapped to indices [1, 2, 3, 4, 5]
    recovered:['I', 'like', 'cats', 'and', 'dogs']
    read I like nothing, which got mapped to indices [1, 2, 6]
    recovered:['I', 'like', 'nothing']
    read I like cats and nothing else, which got mapped to indices [1, 2, 3, 4, 6, 0]
    recovered:['I', 'like', 'cats', 'and', 'nothing', '***UNK***']


A common operation to do with vocabular objects is to replace rare words
with UNKNOWN tokens. We'll convert words that occured less than 2 times.

.. code:: python

    # this is actually a copy operation, because indices change when words are removed from the vocabulary
    vocab = vocab.prune_rares(cutoff=2)
    for s in sents:
        inds = vocab.words2indices(s.split())
        print vocab.indices2words(inds)


.. parsed-literal::

    ['I', 'like', '***UNK***', '***UNK***', '***UNK***']
    ['I', 'like', '***UNK***']
    ['I', 'like', '***UNK***', '***UNK***', '***UNK***', '***UNK***']


Now, we'll convert the entire dataset. The ``convert`` function applies
a transform to the specified field of the dataset. We'll apply a
transform using the vocabulary.

.. code:: python

    from stanza.text.vocab import SennaVocab
    vocab = SennaVocab()
    
    # we'll actually just use the first 200 tokens of the document
    max_len = 200
    train = train.convert({'X': lambda x: x[:max_len]}, in_place=True)
    test = test.convert({'X': lambda x: x[:max_len]}, in_place=True)
        
    # make a backup
    train_orig = train
    test_orig = test
    
    train = train_orig.convert({'X': vocab.update}, in_place=False)
    vocab = vocab.prune_rares(cutoff=3)
    train = train_orig.convert({'X': vocab.words2indices}, in_place=False)
    test = test_orig.convert({'X': vocab.words2indices}, in_place=False)
    pad_index = vocab.add('***PAD***', count=100)
    
    max_len = max([len(x) for x in train.fields['X'] + test.fields['X']])
    
    print 'train: {}, test: {}'.format(len(train), len(test))
    print 'vocab size: {}'.format(vocab)
    print 'sequence max len: {}'.format(max_len)
    print
    print test[:2]


.. parsed-literal::

    train: 751, test: 323
    vocab size: Vocab(4217 words)
    sequence max len: 200
    
    OrderedDict([('Y', [0, 1]), ('X', [[1, 2, 824, 4, 825, 826, 7, 9, 2, 10, 2, 757, 129, 828, 295, 10, 2, 2585, 2586, 302, 19, 2, 831, 252, 832, 240, 25, 2, 1622, 48, 122, 0, 3627, 4, 3628, 3629, 7, 121, 2, 71, 825, 826, 4, 824, 7, 480, 2, 124, 806, 289, 808, 373, 14, 283, 202, 34, 1009, 763, 192, 14, 15, 2096, 3960, 124, 11, 34, 0, 0, 54, 391, 0, 18, 275, 158, 1009, 763, 192, 14, 256, 580, 3960, 124, 389, 232, 0, 46, 71, 528, 187, 2244, 86, 2400, 58, 1725, 0, 1311, 0, 3486, 156, 71, 0, 46, 1, 283, 47, 430, 22, 1010, 4163, 2674, 75, 931, 86, 71, 0, 76, 3935, 46, 2012, 1747, 54, 0, 62, 283, 208, 289, 1290, 331, 275, 825], [1, 2, 0, 9, 2, 236, 14, 34, 0, 0, 19, 2, 608, 25, 2, 2097, 0, 62, 0, 222, 47, 249, 50, 86, 615, 14, 4071, 250, 42, 330, 4, 58, 7, 108, 1319, 236, 4, 289, 2324, 105, 7, 22, 62, 4, 0, 7, 65, 14, 34, 0, 0, 1247, 3553, 31, 1063, 2, 64, 15, 335, 0, 0, 64, 15, 210, 54, 34, 1564, 145, 146, 62, 15, 0, 64, 15, 210, 54, 34, 1564, 145, 343, 15, 343, 146, 62, 15, 0, 64, 201, 426, 58, 717, 0, 14, 1154, 64, 201, 426, 204, 58, 717, 0, 14, 1154, 64, 787, 564, 54, 1154, 101, 0, 89, 1178, 46, 593, 594, 22, 47, 1290, 467, 53, 1292, 2, 18, 17, 47, 1165, 34, 0, 31, 1430, 760, 62, 105, 73, 204, 31, 79, 17, 18, 64, 201, 89, 201, 103, 275, 524, 192, 1407, 22, 47, 179, 1720, 4, 14, 343, 3797, 343, 224, 202, 7, 250, 192, 3450, 1680, 34, 717, 455, 0, 1190, 62, 47, 249, 848, 50, 86, 0, 2766, 847, 0, 550, 86, 343, 553, 330, 65, 278, 343, 46, 816, 22, 289, 1793, 204, 98, 1040, 156, 251, 11, 1443, 225]])])


Training a model
----------------

At this point, you're welcome to use whatever program/model/package you
like to run your experiments. We'll try our hands at Tensorflow. In
particular, we'll define a LSTM classifier.

Model definition
~~~~~~~~~~~~~~~~

We'll define a lookup table, a LSTM, and a linear classifier.

.. code:: python

    import tensorflow as tf    
    from tensorflow.models.rnn import rnn    
    from tensorflow.models.rnn.rnn_cell import LSTMCell
    from stanza.ml.tensorflow_utils import labels_to_onehots
    import numpy as np
    
    np.random.seed(42)      
    embedding_size = 50
    hidden_size = 100
    seq_len = max_len
    vocab_size = len(vocab)
    class_size = len(classes)
    
    # symbolic variable for word indices
    indices = tf.placeholder(tf.int32, [None, seq_len])
    # symbolic variable for labels
    labels = tf.placeholder(tf.float32, [None, class_size])

.. code:: python

    # lookup table
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        E = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="emb")
        embeddings = tf.nn.embedding_lookup(E, indices)
        embeddings_list = [tf.squeeze(t, [1]) for t in tf.split(1, seq_len, embeddings)]

.. code:: python

    # rnn
    cell = LSTMCell(hidden_size, embedding_size)  
    outputs, states = rnn.rnn(cell, embeddings_list, dtype=tf.float32)
    final_output = outputs[-1]

.. code:: python

    # classifier
    def weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
    scores = tf.matmul(final_output, weights((hidden_size, class_size)))

.. code:: python

    # objective
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, labels))

We'll optimize the network via Adam

.. code:: python

    # operations
    train_op = tf.train.AdamOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(scores, 1)

Training
~~~~~~~~

We'll train the network for a fixed number of epochs and then evaluate
on the test set. This is a relatively simple procedure without tuning,
regularization and early stopping.

.. code:: python

    from sklearn.metrics import accuracy_score
    from time import time
    batch_size = 128
    num_epochs = 10
    
    def run_epoch(split, train=False):
        epoch_cost = 0
        epoch_pred = []
        for i in xrange(0, len(split), batch_size):
            batch = split[i: i+batch_size]
            n = len(batch['Y'])
            X = Dataset.pad(batch['X'], pad_index, seq_len)
            Y = np.zeros((n, class_size))
            Y[np.arange(n), np.array(batch['Y'])] = 1
            if train:
                batch_cost, batch_pred, _ = session.run([cost, predict_op, train_op], {indices: X, labels: Y})
            else:
                batch_cost, batch_pred = session.run([cost, predict_op], {indices: X, labels: Y})
            epoch_cost += batch_cost * n
            epoch_pred += batch_pred.flatten().tolist()
        return epoch_cost, epoch_pred
    
    def train_eval(session):
        for epoch in xrange(num_epochs):
            start = time()
            print 'epoch: {}'.format(epoch)
            epoch_cost, epoch_pred = run_epoch(train, True)
            print 'train cost: {}, acc: {}'.format(epoch_cost/len(train), accuracy_score(train.fields['Y'], epoch_pred))
            print 'time elapsed: {}'.format(time() - start)
        
        test_cost, test_pred = run_epoch(test, False)
        print '-' * 20
        print 'test cost: {}, acc: {}'.format(test_cost/len(test), accuracy_score(test.fields['Y'], test_pred))
    
    with tf.Session() as session:
        tf.set_random_seed(123)
        session.run(tf.initialize_all_variables())
        train_eval(session)


.. parsed-literal::

    epoch: 0
    train cost: 0.69150376931, acc: 0.533954727031
    time elapsed: 11.6190190315
    epoch: 1
    train cost: 0.68147453257, acc: 0.587217043941
    time elapsed: 9.51137089729
    epoch: 2
    train cost: 0.662719958632, acc: 0.589880159787
    time elapsed: 9.50747179985
    epoch: 3
    train cost: 0.629683734098, acc: 0.688415446072
    time elapsed: 9.8141450882
    epoch: 4
    train cost: 0.611709104159, acc: 0.709720372836
    time elapsed: 9.49769997597
    epoch: 5
    train cost: 0.582100759651, acc: 0.684420772304
    time elapsed: 10.240678072
    epoch: 6
    train cost: 0.570877154404, acc: 0.737683089214
    time elapsed: 9.94308805466
    epoch: 7
    train cost: 0.564803322447, acc: 0.720372836218
    time elapsed: 9.75270009041
    epoch: 8
    train cost: 0.542043169631, acc: 0.757656458056
    time elapsed: 10.0136928558
    epoch: 9
    train cost: 0.490948782978, acc: 0.78828229028
    time elapsed: 12.185503006
    --------------------
    test cost: 0.591299379758, acc: 0.693498452012


Remember how we used ``SennaVocab``? Let's see what happens if we
preinitialize our embeddings:

.. code:: python

    preinit_op = E.assign(vocab.get_embeddings())
    with tf.Session() as session:
        tf.set_random_seed(123)
        session.run(tf.initialize_all_variables())
        session.run(preinit_op)
        train_eval(session)


.. parsed-literal::

    epoch: 0
    train cost: 0.688563662267, acc: 0.539280958722
    time elapsed: 13.9362518787
    epoch: 1
    train cost: 0.674707842254, acc: 0.584553928096
    time elapsed: 11.8684880733
    epoch: 2
    train cost: 0.663795230312, acc: 0.607190412783
    time elapsed: 11.9107489586
    epoch: 3
    train cost: 0.641969507131, acc: 0.645805592543
    time elapsed: 12.0843448639
    epoch: 4
    train cost: 0.619178395138, acc: 0.660452729694
    time elapsed: 12.1125848293
    epoch: 5
    train cost: 0.591043990636, acc: 0.711051930759
    time elapsed: 12.0441889763
    epoch: 6
    train cost: 0.568309741633, acc: 0.712383488682
    time elapsed: 11.8249480724
    epoch: 7
    train cost: 0.52520389722, acc: 0.772303595206
    time elapsed: 11.7157990932
    epoch: 8
    train cost: 0.501435582949, acc: 0.756324900133
    time elapsed: 11.6853508949
    epoch: 9
    train cost: 0.439889284647, acc: 0.809587217044
    time elapsed: 11.5280079842
    --------------------
    test cost: 0.598627277203, acc: 0.699690402477


