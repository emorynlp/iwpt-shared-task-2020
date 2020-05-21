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


def main():
    cdroot()
    fs = sorted(glob.glob('data/iwpt2020/test-udpipe/*.conllu'))
    fs = [f for f in fs if 'fixed' not in f]
    for idx, f in enumerate(fs):
        langcode = f.split('/')[-2]
        if len(langcode) != 2:
            langcode = os.path.basename(f).split('.')[0]
        if langcode != 'en':
            continue
        print(f'{idx + 1:02d}/{len(fs)} {langcode}')
        with open(f.replace('.conllu', '.fixed.conllu'), 'w') as out:
            for sent in load_conll(f):
                for line in sent.split('\n'):
                    if line.startswith('#') or '-' in line.split('\t')[0]:
                        out.write(line)
                    else:
                        cells = line.split('\t')
                        cells[8] = cells[6] + ':' + cells[7]
                        out.write('\t'.join(cells))
                    out.write('\n')
                out.write('\n')


if __name__ == '__main__':
    main()
