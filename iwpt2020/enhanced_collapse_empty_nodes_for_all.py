# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-26 21:27
import glob

from edparser.metrics.parsing.iwpt20_eval import enhanced_collapse_empty_nodes

from iwpt2020 import cdroot


def main():
    print('Collapse empty nodes for all conllu files ...')
    cdroot()
    conllu_files = glob.glob('data/iwpt2020/**/*.conllu', recursive=True)
    conllu_files = [x for x in conllu_files if 'enhanced_collapse_empty_nodes' not in x]
    for idx, f in enumerate(conllu_files):
        print(f'\r{idx + 1}/{len(conllu_files)} {f}', end='')
        enhanced_collapse_empty_nodes(f, f'{f.replace(".conllu", ".enhanced_collapse_empty_nodes.conllu")}')


if __name__ == '__main__':
    main()
