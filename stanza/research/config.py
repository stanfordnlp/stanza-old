import argparse
import configargparse
import os
import sys
import json
import logfile
import traceback
import StringIO
import contextlib
import __builtin__
from pyhocon import ConfigFactory


class ArgumentParser(configargparse.Parser):
    def convert_setting_to_command_line_arg(self, action, key, value):
        args = []
        if action is None:
            command_line_key = \
                self.get_command_line_key_for_unknown_config_file_setting(key)
        else:
            command_line_key = action.option_strings[-1]

        if isinstance(action, argparse._StoreTrueAction):
            if value is True:
                args.append(command_line_key)
        elif isinstance(action, argparse._StoreFalseAction):
            if value is False:
                args.append(command_line_key)
        elif isinstance(action, argparse._StoreConstAction):
            if value == action.const:
                args.append(command_line_key)
        elif isinstance(action, argparse._CountAction):
            for _ in range(value):
                args.append(command_line_key)
        elif action is not None and value == action.default:
            pass
        elif isinstance(value, list):
            args.append(command_line_key)
            args.extend([str(e) for e in value])
        else:
            args.append(command_line_key)
            args.append(str(value))
        return args


class HoconConfigFileParser(object):
    def parse(self, stream):
        try:
            basedir = os.path.dirname(stream.name)
        except AttributeError:
            basedir = os.getcwd()
        return dict(ConfigFactory.parse_string(stream.read(), basedir=basedir))

    def serialize(self, items):
        return json.dumps(items, sort_keys=True, indent=2, separators=(',', ': '))

    def get_syntax_description(self):
        return ('Config files should use HOCON syntax. HOCON is a superset of '
                'JSON; for more, see '
                '<https://github.com/typesafehub/config/blob/master/HOCON.md>.')


_options_parser = ArgumentParser(conflict_handler='resolve', add_help=False,
                                 config_file_parser=HoconConfigFileParser(),
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_options_parser.add_argument('--run_dir', '-R', type=str, default=None,
                             help='The directory in which to write log files, parameters, etc. '
                                  'Will be created if it does not exist. If None, output files '
                                  'will not be written.')
_options_parser.add_argument('--config', '-C', default=None, is_config_file=True,
                             help='Path to a JSON or HOCON file containing option settings. '
                                  'Can be loaded from the config.json of a previous run to rerun '
                                  'an experiment. If None, only options given as command line '
                                  'arguments will be used.')
_options_parser.add_argument('--overwrite', '-O', action='store_true',
                             help='If True, allow overwriting the contents of the run directory. '
                                  'Otherwise, an error will be raised if the run directory '
                                  'contains a config.json to prevent accidental overwriting. ')


def get_options_parser():
    return _options_parser


_options = None


def options(allow_partial=False, read=False):
    '''
    Get the object containing the values of the parsed command line options.

    :param bool allow_partial: If `True`, ignore unrecognized arguments and allow
        the options to be re-parsed next time `options` is called. This
        also suppresses overwrite checking (the check is performed the first
        time `options` is called with `allow_partial=False`).
    :param bool read: If `True`, do not create or overwrite a `config.json`
        file, and do not check whether such file already exists. Use for scripts
        that read from the run directory rather than/in addition to writing to it.

    :return argparse.Namespace: An object storing the values of the options specified
        to the parser returned by `get_options_parser()`.
    '''
    global _options

    if allow_partial:
        opts, extras = _options_parser.parse_known_args()
        if opts.run_dir:
            mkdirp(opts.run_dir)
        return opts

    if _options is None:
        # Add back in the help option (only show help and quit once arguments are finalized)
        _options_parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                                     help='show this help message and exit')
        _options = _options_parser.parse_args()
        if _options.run_dir:
            mkdirp(_options.run_dir, overwrite=_options.overwrite or read)

        if not read:
            options_dump = vars(_options)
            # People should be able to rerun an experiment with -C config.json safely.
            # Don't include the overwrite option, since using a config from an experiment
            # done with -O should still require passing -O for it to be overwritten again.
            del options_dump['overwrite']
            # And don't write the name of the other config file in this new one! It's
            # probably harmless (config file interpretation can't be chained with the
            # config option), but still confusing.
            del options_dump['config']
            dump_pretty(options_dump, 'config.json')
    return _options


class OverwriteError(Exception):
    pass


def mkdirp(dirname, overwrite=True):
    '''
    Create a directory at the path given by `dirname`, if it doesn't
    already exist. If `overwrite` is False, raise an error when trying
    to create a directory that already has a config.json file in it.
    Otherwise do nothing if the directory already exists. (Note that an
    existing directory without a config.json will not raise an error
    regardless.)

    http://stackoverflow.com/a/14364249/4481448
    '''
    try:
        os.makedirs(dirname)
    except OSError:
        if not os.path.isdir(dirname):
            raise
        config_path = os.path.join(dirname, 'config.json')
        if not overwrite and os.path.lexists(config_path):
            raise OverwriteError('%s exists and already contains a config.json. To allow '
                                 'overwriting, pass the -O/--overwrite option.' % dirname)


def get_file_path(filename):
    opts = options(allow_partial=True)
    if not opts.run_dir:
        return None
    return os.path.join(opts.run_dir, filename)


def open(filename, *args, **kwargs):
    file_path = get_file_path(filename)
    if not file_path:
        # create a dummy file because we don't have a run dir
        return contextlib.closing(StringIO.StringIO())
    return __builtin__.open(file_path, *args, **kwargs)


def boolean(arg):
    """Convert a string to a bool treating 'false' and 'no' as False."""
    if arg in ('true', 'True', 'yes', '1', 1):
        return True
    elif arg in ('false', 'False', 'no', '0', 0):
        return False
    else:
        raise argparse.ArgumentTypeError(
            'could not interpret "%s" as true or false' % (arg,))


def redirect_output():
    outfile = get_file_path('stdout.log')
    if outfile is None:
        return
    logfile.log_stdout_to(outfile)
    logfile.log_stderr_to(get_file_path('stderr.log'))


def dump(data, filename, lines=False, *args, **kwargs):
    try:
        with open(filename, 'w') as outfile:
            if lines:
                for item in data:
                    json.dump(item, outfile, *args, **kwargs)
                    outfile.write('\n')
            else:
                json.dump(data, outfile, *args, **kwargs)
    except IOError:
        traceback.print_exc()
        print >>sys.stderr, 'Unable to write %s' % filename
    except TypeError:
        traceback.print_exc()
        print >>sys.stderr, 'Unable to write %s' % filename


def dump_pretty(data, filename):
    dump(data, filename,
         sort_keys=True, indent=2, separators=(',', ': '))
