#!/usr/bin/env python
'''
Print the name of a device to use, either 'cpu' or 'gpu0', 'gpu1',...
The least-used GPU with usage under the constant threshold will be chosen;
ties are broken randomly.

Can be called from the shell, with no arguments:

  $ python pick_gpu.py
  gpu0

Warning: This is hacky and brittle, and can break if nvidia-smi changes
in the way it formats its output.
'''
__author__ = 'sbowman@stanford.edu, wmonroe4@stanford.edu'

import subprocess
import sys
import random
from collections import namedtuple


USAGE_THRESHOLD = 0.8

Usage = namedtuple('Usage', 'fan,mem,cpu')


def best_gpu(max_usage=USAGE_THRESHOLD, verbose=False):
    '''
    Return the name of a device to use, either 'cpu' or 'gpu0', 'gpu1',...
    The least-used GPU with usage under the constant threshold will be chosen;
    ties are broken randomly.
    '''
    try:
        proc = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        output, error = proc.communicate()
        if error:
            raise Exception(error)
    except Exception, e:
        sys.stderr.write("Couldn't run nvidia-smi to find best GPU, using CPU: %s\n" % str(e))
        sys.stderr.write("(This is normal if you have no GPU or haven't configured CUDA.)\n")
        return "cpu"

    usages = parse_output(output)

    pct_usage = [max(u.mem, cpu_backoff(u)) for u in usages]
    max_usage = min(max_usage, min(pct_usage))

    open_gpus = [index for index, usage in enumerate(usages)
                 if max(usage.mem, cpu_backoff(usage)) <= max_usage]
    if verbose:
        print('Best GPUs:')
        for index in open_gpus:
            print('%d: %s fan, %s mem, %s cpu' %
                  (index, format_percent(usages[index].fan),
                   format_percent(usages[index].mem),
                   format_percent(usages[index].cpu)))

    if open_gpus:
        result = "gpu" + str(random.choice(open_gpus))
    else:
        result = "cpu"

    if verbose:
        print('Chosen: ' + result)
    return result


def parse_output(output):
    start = output.index('|===')
    end = output.index('\n   ')
    lines = output[start:end].split('\n')[2::3]
    fields = [line.split() for line in lines]

    fan_fields = [line[1] for line in fields]
    mem_used_fields = [line[8] for line in fields]
    total_mem_fields = [line[10] for line in fields]
    cpu_fields = [line[12] for line in fields]

    fan_amts = [parse_percent(f) for f in fan_fields]
    mem_used_amts = [parse_bytes(f) for f in mem_used_fields]
    total_mem_amts = [parse_bytes(f) for f in total_mem_fields]
    cpu_amts = [parse_percent(f) for f in cpu_fields]

    pct_mem_used = [(float(usage_amt) / float(total)
                     if None not in (usage_amt, total)
                     else None)
                    for (usage_amt, total) in zip(mem_used_amts, total_mem_amts)]

    return [Usage(fan, mem, cpu)
            for (fan, mem, cpu) in zip(fan_amts, pct_mem_used, cpu_amts)]


def parse_percent(field):
    try:
        if field.endswith('%'):
            return float(field[:-1])
        else:
            return float(field)
    except ValueError:
        return None


def parse_bytes(field):
    '''
    >>> parse_bytes('24B')
    24.0
    >>> parse_bytes('4MiB')
    4194304.0
    '''
    if field[-1] in 'bB':
        field = field[:-1]

    try:
        for i, prefix in enumerate('KMGTPEZ'):
            if field.endswith(prefix + 'i'):
                factor = 2 ** (10 * (i + 1))
                return float(field[:-2]) * factor

        return float(field)
    except ValueError:
        return None


def cpu_backoff(u):
    if u.cpu is not None:
        return u.cpu
    elif u.fan is not None:
        return u.fan
    else:
        return 0.0


def format_percent(p):
    if p is None:
        return 'N/A'
    else:
        return '%f%%' % p


def bind_theano(device=None, max_usage=USAGE_THRESHOLD, verbose=True):
    '''
    Initialize Theano to use a certain device. If `device` is None (the
    default), use the device returned by calling `best_gpu`
    with the same parameters.

    This needs to be called *before* importing Theano. Currently (Dec 2015)
    Theano has no way of switching devices after it is bound (which happens
    on import).
    '''
    if device is None:
        device = best_gpu(max_usage, verbose=verbose)
    if device and device != 'cpu':
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)


__all__ = [
    'best_gpu',
    'bind_theano',
]


if __name__ == '__main__':
    print(best_gpu())
