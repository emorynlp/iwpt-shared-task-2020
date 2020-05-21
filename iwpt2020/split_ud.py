# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-17 11:38
import math

from iwpt2020 import cdroot


def main():
    cdroot()
    input_file = 'data/parsing/ko_penn-ud-revised/ko_penn-ud-revised.conllu'
    train_file = input_file.replace('.conllu', '.train.conllu')
    dev_file = input_file.replace('.conllu', '.dev.conllu')
    test_file = input_file.replace('.conllu', '.test.conllu')
    with open(input_file) as src, open(train_file, 'w') as train, open(dev_file, 'w') as dev, open(test_file,
                                                                                                   'w') as test:
        everything = src.read()
        sents = everything.split('\n\n')
        fileid_sents = {}
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            fileid = sent.split('\n')[0].split()[3].split('.')[0]
            sents_in_file = fileid_sents.get(fileid, None)
            if not sents_in_file:
                sents_in_file = []
                fileid_sents[fileid] = sents_in_file
            sents_in_file.append(sent)
        counts = [0] * 3
        for each_file, sents_in_file in fileid_sents.items():
            total = len(sents_in_file)
            t = math.ceil(total * .79)
            d = math.floor(total * .9105)
            for idx, (f, ss) in enumerate(
                    zip([train, dev, test], [sents_in_file[:t], sents_in_file[t:d], sents_in_file[d:]])):
                if ss:
                    f.write('\n\n'.join(ss))
                    f.write('\n\n')
                    counts[idx] += len(ss)
        print(counts)


if __name__ == '__main__':
    main()
