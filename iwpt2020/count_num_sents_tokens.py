# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-09 07:45

import glob
import os

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


trn = {}
dev = {}
tst = {}


def run(lang):
    dataset = f'data/iwpt2020/train-dev-combined/{lang}'
    trnfile = f'{dataset}/train.conllu'
    devfile = f'{dataset}/dev.conllu'
    testfile = f'data/iwpt2020/test-udpipe/{lang}.conllu'
    for stat, f in zip([trn, dev, tst], [trnfile, devfile, testfile]):
        data = load_conll(f)
        if 'num_sents' not in stat:
            stat['num_sents'] = {}
        if 'num_tokens' not in stat:
            stat['num_tokens'] = {}
        stat['num_sents'][lang] = len(data)
        stat['num_tokens'][lang] = sum(len([y for y in x.split('\n') if y.strip() and not y.startswith('#')]) for x in data)


def main():
    cdroot()
    testfiles = sorted(glob.glob('data/iwpt2020/test-blind/*.txt'))
    for idx, txt in enumerate(testfiles):
        basename = os.path.basename(txt)
        langcode = basename.split('.')[0]
        print(f'{idx + 1:02d}/{len(testfiles)} {basename}')
        run(langcode)
    for stat, split in zip([trn, dev, tst], ['trn', 'dev', 'tst']):
        print(split)
        for k, v in stat.items():
            print(f'{k}\t', end='')
            for lang, num in v.items():
                print(lang, end='\t')
            print()
            for lang, num in v.items():
                print(num, end='\t')
            print()
        print()


if __name__ == '__main__':
    main()
