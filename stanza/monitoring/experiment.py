"""
Utilities useful for experiment setup
"""
__author__ = 'victor'
import json


class AttrDict(dict):
    """
    A dictionary object which keys can be referenced via attributes.

    Example:

    .. code-block:: python

        d = AttriDict(foo=1, bar='cool')
        print(d)
        print(d['foo'])
        print(d.foo)
        print(d.bar)
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def save(self, fname):
        """ Saves the dictionary in json format
        :param fname: file to save to
        """
        with open(fname, 'wb') as f:
            json.dump(self, f)

    @classmethod
    def load(cls, fname):
        """ Loads the dictionary from json file
        :param fname: file to load from
        :return: loaded dictionary
        """
        with open(fname) as f:
            return Config(**json.load(f))
