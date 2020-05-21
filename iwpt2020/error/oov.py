# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-06 23:16
from edparser.utils.io_util import load_pickle, save_pickle
from iwpt2020 import cdroot
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


iv = set()

for each in load_conll('data/iwpt2020/train-dev-combined/en/train.conllu'):
    for cell in load_sent(each):
        form = cell.split('\t')[1].lower()
        iv.add(form)


def calc_f1(path):
    correct = 0
    ngold = 0
    npred = 0
    for gold, pred in zip(load_conll(gold_file), load_conll(path)):
        gt = set()
        pt = set()
        for gold, pred in zip(load_sent(gold), load_sent(pred)):
            gf = gold.split('\t')[1].lower()
            pf = pred.split('\t')[1].lower()
            if gf in iv:
                continue
            idx = gold.split('\t')[0]
            for rel in gold.split('\t')[8].split('|'):
                gt.add((idx,) + tuple(rel.split(':')))
            for rel in pred.split('\t')[8].split('|'):
                pt.add((idx,) + tuple(rel.split(':')))

        ngold += len(gt)
        npred += len(pt)
        correct += len(gt & pt)
    p = correct / npred
    r = correct / ngold
    f1 = 2 * p * r / (p + r)
    return f1

fig, ax = plt.subplots()
ind = np.arange(3)
width = 0.35
try:
    cache = load_pickle('cache_f1.pkl')
except FileNotFoundError:
    cache = {}
for lang in ['mbert', 'bert']:
    f1s = []
    for model, color in zip(['dep', 'sdp', 'ens'], 'rgb'):
        key = f'{lang}-{model}'
        if key in cache:
            f1 = cache[key]
        else:
            pred_file = template.replace('bert', lang).replace('dep', model)
            f1 = calc_f1(pred_file)
            cache[key] = f1
        f1s.append(f1)
        print(key)
    ax.bar(ind + (width if lang == 'bert' else 0), f1s, width, label='multilingual' if lang.startswith('m') else 'language-specific')

save_pickle(cache, 'cache_f1.pkl')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(['DTP', 'DGP', 'ENS'])
plt.ylabel('ELAS of OOV')
ax.legend()
plt.savefig('oov.pdf')
plt.show()
