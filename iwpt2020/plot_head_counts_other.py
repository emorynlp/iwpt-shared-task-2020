# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-26 15:34
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from edparser.components.parsers.conll import CoNLLSentence
from iwpt2020 import cdroot


def plot_bar(some_counter, xlabel=None, ylabel=None):
    labels, values = zip(*sorted(some_counter.items()))

    indexes = np.arange(len(labels))
    width = 0.3

    plt.bar(indexes, values, width)
    plt.xticks(indexes, labels)
    for i, v in enumerate(values):
        plt.text(indexes[i] - int(np.log10(v)) * 0.2 * width, v + 0.01 * max(values), str(v))
    if len(some_counter) > 10:
        plt.xticks(rotation=45)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


def main():
    cdroot()
    dataset = CoNLLSentence.from_file(
        'data/iwpt2020/train-dev/UD_Arabic-PADT/ar_padt-ud-train.enhanced_collapse_empty_nodes.conllu', conllu=True)
    data = Counter()
    for each in dataset:
        for word in each:
            data[len(word.head)] += 1
    plot_bar(data, 'num of heads', 'num of samples')


if __name__ == '__main__':
    main()
