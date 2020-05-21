# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-22 21:09
import glob
import os
from shutil import copyfile

from edparser.metrics.parsing.iwpt20_eval import conllu_quick_fix, remove_complete_edges, remove_collapse_edges
from iwpt2020 import cdroot


def main():
    cdroot()
    submission = 'data/model/iwpt2020/emorynlp-submission'
    os.makedirs(submission, exist_ok=True)
    fs = sorted(glob.glob('data/iwpt2020/test-blind/*.txt'))
    for idx, txt in enumerate(fs):
        basename = os.path.basename(txt)
        langcode = basename.split('.')[0]
        print(f'{idx + 1:02d}/{len(fs)} {basename}')
        src = f'data/model/iwpt2020/{langcode}/{langcode}.conllu'
        dst = f'{submission}/{langcode}.conllu'
        tmp = f'/home/hhe43/tmp/{langcode}.conllu'
        copyfile(src, tmp)
        remove_complete_edges(tmp, tmp)
        remove_collapse_edges(tmp, tmp)
        src = tmp
        conllu_quick_fix(src, dst)


if __name__ == '__main__':
    main()
