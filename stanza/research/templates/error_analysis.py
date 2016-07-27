import glob
import json
import Levenshtein as lev
import numpy as np
import os
import warnings
from collections import namedtuple

from stanza.util.unicode import uprint
from stanza.research import config


parser = config.get_options_parser()
parser.add_argument('--max_examples', type=int, default=100,
                    help='The maximum number of examples to display in error analysis.')
parser.add_argument('--html', type=config.boolean, default=False,
                    help='If true, output errors in HTML.')

Output = namedtuple('Output', 'config,results,data,scores,predictions')


COLORS = ['black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white']
HTML = ['Black', 'DarkRed', 'DarkGreen', 'Olive', 'Blue', 'Purple', 'DarkCyan', 'White']


def wrap_color_html(s, color):
    code = COLORS.index(color)
    if code == -1:
        html_color = color
    else:
        html_color = HTML[code]
    return '<span style="color:%s;font-weight:bold">%s</span>' % (html_color, s)


def wrap_color_shell(s, color):
    code = COLORS.index(color)
    if code == -1:
        raise ValueError('unrecognized color: ' + color)
    return '\033[1;3%dm%s\033[0m' % (code, s)


def highlight(text, positions, color, html=False):
    chars = []
    wrap_color = wrap_color_html if html else wrap_color_shell
    for i, c in enumerate(text):
        if i in positions:
            chars.append(wrap_color(c, color))
        else:
            chars.append(c)
    return u''.join(chars)


def print_error_analysis():
    options = config.options(read=True)
    output = get_output(options.run_dir, 'eval')
    errors = [(inst['input'], pred, inst['output'])
              for inst, pred in zip(output.data, output.predictions)
              if inst['output'] != pred]
    if 0 < options.max_examples < len(errors):
        indices = np.random.choice(np.arange(len(errors)), size=options.max_examples, replace=False)
    else:
        indices = range(len(errors))

    if options.html:
        print('<!DOCTYPE html>')
        print('<html><head><title>Error analysis</title><meta charset="utf-8" /></head><body>')
    for i in indices:
        inp, pred, gold = [unicode(s).strip() for s in errors[i]]
        editops = lev.editops(gold, pred)
        print_visualization(inp, pred, gold, editops, html=options.html)
    if options.html:
        print('</body></html>')


def print_visualization(input_seq, pred_output_seq,
                        gold_output_seq, editops, html=False):
    gold_highlights = []
    pred_highlights = []
    for optype, gold_idx, pred_idx in editops:
        gold_highlights.append(gold_idx)
        pred_highlights.append(pred_idx)

    input_seq = highlight(input_seq, pred_highlights, 'cyan', html=html)
    pred_output_seq = highlight(pred_output_seq, pred_highlights, 'red', html=html)
    gold_output_seq = highlight(gold_output_seq, gold_highlights, 'yellow', html=html)

    if html:
        print('<p>')
        br = u' <br />'
    else:
        br = u''
    uprint(input_seq + br)
    uprint(pred_output_seq + br)
    uprint(gold_output_seq)
    if html:
        print('</p>')
    print('')


def get_output(run_dir, split):
    config_dict = load_dict(os.path.join(run_dir, 'config.json'))

    results = {}
    for filename in glob.glob(os.path.join(run_dir, 'results.*.json')):
        results.update(load_dict(filename))

    data = load_dataset(os.path.join(run_dir, 'data.%s.jsons' % split))
    scores = load_dataset(os.path.join(run_dir, 'scores.%s.jsons' % split))
    predictions = load_dataset(os.path.join(run_dir, 'predictions.%s.jsons' % split))
    return Output(config_dict, results, data, scores, predictions)


def load_dict(filename):
    try:
        with open(filename) as infile:
            return json.load(infile)
    except IOError, e:
        warnings.warn(str(e))
        return {'error.message.value': str(e)}


def load_dataset(filename, transform_func=(lambda x: x)):
    try:
        dataset = []
        with open(filename) as infile:
            for line in infile:
                js = json.loads(line.strip())
                dataset.append(transform_func(js))
        return dataset
    except IOError, e:
        warnings.warn(str(e))
        return [{'error': str(e)}]


if __name__ == '__main__':
    print_error_analysis()
