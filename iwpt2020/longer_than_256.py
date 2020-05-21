# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-09 07:45

import glob
import os
from collections import Counter

from edparser.components.parsers.conll import read_conll
from edparser.layers.transformers import AutoTokenizer, AutoConfig
from edparser.layers.transformers.utils import config_is, convert_examples_to_features
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


def len_of_sent(sent, config, tokenizer, max_seq_length=1e8):
    # Transformer tokenizing
    xlnet = config_is(config, 'xlnet')
    roberta = config_is(config, 'roberta')
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    sent = sent[1:]  # remove <root> use [CLS] instead
    pad_label_idx = 0
    input_ids, input_mask, segment_ids, prefix_mask = \
        convert_examples_to_features(sent,
                                     max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=xlnet,
                                     # xlnet has a cls token at the end
                                     cls_token=cls_token,
                                     cls_token_segment_id=2 if xlnet else 0,
                                     sep_token=sep_token,
                                     sep_token_extra=roberta,
                                     # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                     pad_on_left=xlnet,
                                     # pad on the left for xlnet
                                     pad_token_id=pad_token_id,
                                     pad_token_segment_id=4 if xlnet else 0,
                                     pad_token_label_id=pad_label_idx,
                                     do_padding=False)
    return len(input_ids)


def main():
    cdroot()
    fs = sorted(glob.glob('data/iwpt2020/test-udpipe/??.conllu'))
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    config = AutoConfig.from_pretrained(pretrained)
    stat = Counter()
    for idx, f in enumerate(fs):
        langcode = f.split('/')[-2]
        if len(langcode) != 2:
            langcode = os.path.basename(f).split('.')[0]
        # if langcode != 'fi':
        #     continue
        print(f'{idx + 1:02d}/{len(fs)} {langcode}')
        total = 0
        for sent in read_conll(f):
            words = [x[1] for x in sent]
            length = len_of_sent(words, config, tokenizer)
            if length > 256:
                stat[langcode] += 1
            total += 1
        stat[langcode] += 0
        # stat[langcode] /= total
        # stat[langcode] *= 100
    for k, v in sorted(stat.items()):
        print(k, end='\t')
    print()
    for k, v in sorted(stat.items()):
        print(v, end='\t')
    # plot_bar(stat, 'language', 'percentage of discarded long sentences', sort_y=True, save='long_sents.pdf', ann_y=False)


if __name__ == '__main__':
    main()
