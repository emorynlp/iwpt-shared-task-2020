# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-09 07:45

import glob
import os

from edparser.components.parsers.conll import CoNLL_Transformer_Transform
from edparser.layers.transformers import AutoTokenizer, AutoConfig
from edparser.utils.tf_util import size_of_dataset
from iwpt2020 import cdroot


def load_conll(path):
    with open(path) as src:
        text = src.read()
        sents = text.split('\n\n')
        sents = [x for x in sents if x.strip()]
        return sents


def combine(folders, file, out):
    sents = []
    for f in folders:
        f = glob.glob(f'{f}/*{file}')
        if not f:
            continue
        assert len(f) == 1
        f = f[0]
        sents += load_conll(f)
    with open(out, 'w') as out:
        out.write('\n\n'.join(sents))
        out.write('\n\n')


def run(lang):
    dataset = f'data/iwpt2020/train-dev-combined/{lang}'
    trnfile = f'{dataset}/train.short.conllu'
    devfile = f'{dataset}/dev.short.conllu'
    testfile = f'data/iwpt2020/test-udpipe/{lang}.fixed.short.conllu'
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    config = AutoConfig.from_pretrained(pretrained)
    dep = CoNLL_Transformer_Transform()
    dep.tokenizer = tokenizer
    dep.transformer_config = config
    sdp = CoNLL_Transformer_Transform(graph=True)
    sdp.tokenizer = tokenizer
    sdp.transformer_config = config
    for t in [dep, sdp]:
        for idx, f in enumerate([trnfile, devfile, testfile]):
            if not idx:
                t.fit(f)
            if 'test' not in f:
                continue
            print(f'{f} {size_of_dataset(t.file_to_dataset(f))}')


def main():
    cdroot()
    testfiles = sorted(glob.glob('data/iwpt2020/test-blind/*.txt'))
    for idx, txt in enumerate(testfiles):
        basename = os.path.basename(txt)
        langcode = basename.split('.')[0]
        print(f'{idx + 1:02d}/{len(testfiles)} {basename}')
        if idx + 1 != 3:
            continue
        run(langcode)


if __name__ == '__main__':
    main()
