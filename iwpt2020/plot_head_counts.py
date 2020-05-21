# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-26 15:34
import glob
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from edparser.components.parsers.conll import CoNLLSentence
from iwpt2020 import cdroot


def plot_bar(some_counter, xlabel=None, ylabel=None, sort_y=False, ann_y=True, save=None):
    labels, values = zip(*sorted(some_counter.items(), key=(lambda x: x[1]) if sort_y else None))

    indexes = np.arange(len(labels))
    width = 0.3

    plt.bar(indexes, values, width)
    plt.xticks(indexes, labels)
    if ann_y:
        for i, v in enumerate(values):
            plt.text(indexes[i] - int(np.log10(v)) * 0.2 * width, v + 0.01 * max(values), str(v))
    if len(some_counter) > 10 and all(len(x) > 2 for x in labels):
        plt.xticks(rotation=45)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if save:
        plt.savefig(save)
    plt.show()


def main():
    cdroot()
    fs = sorted(glob.glob('data/iwpt2020/test-blind/*.txt'))
    num_multi_heads = Counter()
    for idx, txt in enumerate(fs):
        basename = os.path.basename(txt)
        langcode = basename.split('.')[0]
        print(f'{idx + 1:02d}/{len(fs)} {basename}')
        stat = head_stat(langcode)
        num_multi_heads[langcode] = (sum(stat.values()) - stat[1]) / sum(stat.values())
        # plot_bar(stat, 'num of heads', 'num of tokens')
    plot_bar(num_multi_heads, 'language', '# tokens with multiple heads', ann_y=False, sort_y=True, save='heads.pdf')


def head_stat(lang):
    trees = CoNLLSentence.from_file(f'data/iwpt2020/train-dev-combined/{lang}/train.conllu', conllu=True)
    stat = Counter()
    for each in trees:
        for word in each:
            heads = word.deps.split('|')
            stat[len(heads)] += 1
    return stat


if __name__ == '__main__':
    main()
