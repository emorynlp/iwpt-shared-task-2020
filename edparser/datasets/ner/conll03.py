# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-06 15:31
from edparser.common.dataset import TransformDataset
from edparser.utils.io_util import get_resource, generator_words_tags_from_tsv

CONLL03_EN_TRAIN = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.train.tsv'
CONLL03_EN_VALID = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.dev.tsv'
CONLL03_EN_TEST = 'https://file.hankcs.com/corpus/conll03_en_iobes.zip#eng.test.tsv'


class TSVTaggingDataset(TransformDataset):
    def load_file(self, filepath):
        filepath = get_resource(filepath)
        for words, tags in generator_words_tags_from_tsv(filepath, lower=False):
            yield {'word': words, 'tag': tags}


def main():
    dataset = TSVTaggingDataset(CONLL03_EN_VALID)
    print(dataset[3])


if __name__ == '__main__':
    main()
