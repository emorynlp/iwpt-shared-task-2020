# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-17 11:38

from iwpt2020 import cdroot
from iwpt2020 import load_conll


def main():
    cdroot()
    input_file = 'data/parsing/ko_penn-ud-revised/ko_penn-ud-revised.conllu'
    train_file = input_file.replace('.conllu', '.train.conllu')
    dev_file = input_file.replace('.conllu', '.dev.conllu')
    test_file = input_file.replace('.conllu', '.test.conllu')
    num_tokens = {}
    num_sents = {}
    for split in 'train', 'dev', 'test':
        num_tokens[split] = 0
        num_sents[split] = 0
        file = input_file.replace('.conllu', f'.{split}.conllu')
        sents = load_conll(file)
        for each in sents:
            if not each.strip():
                continue
            num_sents[split] += 1
            for line in each.split('\n'):
                if line.startswith('#') or len(line.split('\t')) != 10:
                    continue
                num_tokens[split] += 1
    print(f'# of sents: {num_sents}')
    print(f'# of tokens: {num_tokens}')


if __name__ == '__main__':
    main()
