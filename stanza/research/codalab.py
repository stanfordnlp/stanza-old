"""Tools for working with CodaLab."""
import os
import tempfile
import subprocess
import cPickle as pickle
from os.path import abspath
from os.path import dirname

import matplotlib.image as mpimg
import json
import sys
import platform
from contextlib import contextmanager
import shutil

__author__ = 'kelvinguu'


# need to be specified by user
worksheet = None
site = None

# http://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
def shell(cmd, verbose=False, debug=False):
    if verbose:
        print cmd

    if debug:
        return  # don't actually execute command

    output = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    for c in iter(lambda: process.stdout.read(1), ''):
        output.append(c)
        if verbose:
            sys.stdout.write(c)
            sys.stdout.flush()

    status = process.wait()
    if status != 0:
        raise RuntimeError('Error, exit code: {}'.format(status))

    # TODO: make sure we get all output
    return ''.join(output)


def get_uuids():
    """List all bundle UUIDs in the worksheet."""
    result = shell('cl ls -w {} -u'.format(worksheet))
    uuids = result.split('\n')
    uuids = uuids[1:-1]  # trim non uuids
    return uuids


@contextmanager
def open_file(uuid, path):
     """Get the raw file content within a particular bundle at a particular path.

     Path have no leading slash.
     """
     # create temporary file just so we can get an unused file path
     f = tempfile.NamedTemporaryFile()
     f.close()  # close and delete right away
     fname = f.name

     # download file to temporary path
     cmd ='cl down -o {} -w {} {}/{}'.format(fname, worksheet, uuid, path)
     try:
        shell(cmd)
     except RuntimeError:
         try:
            os.remove(fname)  # if file exists, remove it
         except OSError:
             pass
         raise IOError('Failed to open file {}/{}'.format(uuid, path))

     f = open(fname)
     yield f
     f.close()
     os.remove(fname)  # delete temp file


class Bundle(object):
    def __init__(self, uuid):
        self.uuid = uuid

    def __getattr__(self, item):
        """
        Load attributes: history, meta on demand
        """
        if item == 'history':
            try:
                with open_file(self.uuid, 'history.cpkl') as f:
                    value = pickle.load(f)
            except IOError:
                value = {}

        elif item == 'meta':
            try:
                with open_file(self.uuid, 'meta.json') as f:
                    value = json.load(f)
            except IOError:
                value = {}

            # load codalab info
            fields = ('uuid', 'name', 'bundle_type', 'state', 'time', 'remote')
            cmd = 'cl info -w {} -f {} {}'.format(worksheet, ','.join(fields), self.uuid)
            result = shell(cmd)
            info = dict(zip(fields, result.split()))
            value.update(info)

        elif item in ('stderr', 'stdout'):
            with open_file(self.uuid, item) as f:
                value = f.read()

        else:
            raise AttributeError(item)

        self.__setattr__(item, value)
        return value

    def __repr__(self):
        return self.uuid

    def load_img(self, img_path):
        """
        Return an image object that can be immediately plotted with matplotlib
        """
        with open_file(self.uuid, img_path) as f:
            return mpimg.imread(f)


def download_logs(bundle, log_dir):
    if bundle.meta['bundle_type'] != 'run' or bundle.meta['state'] == 'queued':
        print 'Skipped {}\n'.format(bundle.uuid)
        return

    if isinstance(bundle, str):
        bundle = Bundle(bundle)

    uuid = bundle.uuid
    name = bundle.meta['name']
    log_path = os.path.join(log_dir, '{}_{}'.format(name, uuid))

    cmd ='cl down -o {} -w {} {}/logs'.format(log_path, worksheet, uuid)

    print uuid
    try:
        shell(cmd, verbose=True)
    except RuntimeError:
        print 'Failed to download', bundle.uuid
    print


def report(render, uuids=None, reverse=True, limit=None):
    if uuids is None:
        uuids = get_uuids()

    if reverse:
        uuids = uuids[::-1]

    if limit is not None:
        uuids = uuids[:limit]

    for uuid in uuids:
        bundle = Bundle(uuid)
        try:
            render(bundle)
        except Exception:
            print 'Failed to render', bundle.uuid


def monitor_jobs(logdir, uuids=None, reverse=True, limit=None):
    if os.path.exists(logdir):
        delete = raw_input('Overwrite existing logdir? ({})'.format(logdir))
        if delete == 'y':
            shutil.rmtree(logdir)
            os.makedirs(logdir)
    else:
        os.makedirs(logdir)
        print 'Using logdir:', logdir

    report(lambda bd: download_logs(bd, logdir), uuids, reverse, limit)


def tensorboard(logdir):
    print 'Run this in bash:'
    shell('tensorboard --logdir={}'.format(logdir), verbose=True, debug=True)
    print '\nGo to TensorBoard: http://localhost:6006/'


def add_to_sys_path(path):
    """Add a path to the system PATH."""
    sys.path.insert(0, path)


def configure_matplotlib():
    """Set Matplotlib backend to 'Agg', which is necessary on CodaLab docker image."""
    import warnings
    import matplotlib
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        matplotlib.use('Agg')  # needed when running from server


def in_codalab():
    """Check if we are running inside CodaLab Docker container or not."""
    # TODO: below is a total hack. If the OS is not a Mac, we assume we're on CodaLab.
    return platform.system() != 'Darwin'


def launch_job(job_name, cmd=None,
               code_dir=None, excludes='*.ipynb .git .ipynb_checkpoints', dependencies=tuple(),
               queue='john', image='codalab/python', memory='18g',
               debug=False, tail=False):
    """Launch a job on CodaLab (optionally upload code that the job depends on).

    Args:
        job_name: name of the job
        cmd: command to execute
        code_dir: path to code folder. If None, no code is uploaded.
        excludes: file types to exclude from the upload
        dependencies: list of other bundles that we depend on
        debug: if True, prints SSH commands, but does not execute them
        tail: show the streaming output returned by CodaLab once it launches the job
    """
    print 'Remember to set up SSH tunnel and LOG IN through the command line before calling this.'

    def execute(cmd):
        return shell(cmd, verbose=True, debug=debug)

    if code_dir:
        execute('cl up -n code -w {} {} -x {}'.format(worksheet, code_dir, excludes))

    options = '-v -n {} -w {} --request-queue {} --request-docker-image {} --request-memory {}'.format(
        job_name, worksheet, queue, image, memory)
    dep_str = ' '.join(['{0}:{0}'.format(dep) for dep in dependencies])
    cmd = "cl run {} {} '{}'".format(options, dep_str, cmd)
    if tail:
        cmd += ' -t'
    execute(cmd)