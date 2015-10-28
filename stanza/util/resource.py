__author__ = 'victor'
import os
import stanza
import requests

def get_from_url(url):
    return requests.get(url).content

def get_data_or_download(dir_name, file_name, url=''):
    dir = os.path.join(stanza.DATA_DIR, dir_name)
    fname = os.path.join(dir, file_name)
    if not os.path.isdir(dir):
        assert url, 'Could not locate data {}, and url was not specified. Cannot retrieve data.'.format(dir)
        os.makedirs(dir)
    if not os.path.isfile(fname):
        assert url, 'Could not locate data {}, and url was not specified. Cannot retrieve data.'.format(fname)
        print('downloading from {}'.format(url))
        with open(fname, 'wb') as f:
            f.write(get_from_url(url))
    return fname
