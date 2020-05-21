# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-06 23:16
from edparser.utils.io_util import load_pickle, save_pickle
from iwpt2020 import cdroot
from edparser.metrics.parsing.iwpt20_eval import evaluate
import numpy as np
import matplotlib.pyplot as plt

cdroot()

gold_file = 'data/iwpt2020/test-udpipe/en.fixed.conllu'

template = 'data/model/iwpt2020/bert/dep/en.conllu'


def load_conll(path):
    with open(path) as src:
        text = src.read()
        sents = text.split('\n\n')
        sents = [x for x in sents if x.strip()]
        return sents


def load_sent(text: str):
    return [x for x in text.split('\n') if not x.startswith('#')]


def limit_len(file, max_len):
    sents = load_conll(file)
    outfile = file.replace('.conllu', '.tmp.conllu')
    with open(outfile, 'w') as out:
        for each in sents:
            sent = load_sent(each)
            sent = [x for x in sent if '.' not in x.split()[0]]
            if len(sent) > max_len:
                continue
            out.write(f'{each}\n\n')
    return outfile


# evaluate(gold_file, 'data/model/iwpt2020/bert/sdp/en.conllu', do_enhanced_collapse_empty_nodes=True, do_copy_cols=False)

try:
    cache = load_pickle('cache.pkl')
except FileNotFoundError:
    cache = {}
for lang in ['mbert', 'bert']:
    for model, color in zip(['dep', 'sdp', 'ens'], 'rgb'):
        key = f'{lang}-{model}'
        if key in cache:
            xs, ys = cache[key]
        else:
            pred_file = template.replace('bert', lang).replace('dep', model)
            xs = np.arange(5, 50, 5)
            ys = [evaluate(limit_len(gold_file, l), limit_len(pred_file, l), do_copy_cols=False,
                           do_enhanced_collapse_empty_nodes=True)['ELAS'].f1 for l in xs]
            cache[key] = (xs, ys)
        plt.plot(xs, ys, label=key.replace('mbert', 'multilingual').replace('bert', 'language-specific').replace('dep',
                                                                                                                 'DTP').replace(
            'sdp', 'DGP').replace('ens', 'ENS')
                 , color=color, linestyle='-' if lang == 'mbert' else '--')
        print(key)
save_pickle(cache, 'cache.pkl')
plt.xlabel('sentence length')
plt.ylabel('ELAS')
plt.legend()
plt.savefig('sent_len.pdf')
plt.show()
