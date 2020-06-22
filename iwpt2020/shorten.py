# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-09 07:45

import glob
import os

from edparser.components.parsers.conll import read_conll
from edparser.layers.transformers.tf_imports import AutoTokenizer, AutoConfig
from edparser.layers.transformers.utils import config_is, convert_examples_to_features
from edparser.utils.io_util import save_json
from iwpt2020 import cdroot


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


def sent_str(sent: list):
    return '\n'.join('\t'.join(str(x) for x in w) for w in sent)


def main():
    cdroot()
    files = sorted(x for x in glob.glob('data/iwpt2020/train-dev-combined/**/train.conllu') if 'short' not in x)
    shorten(files)
    files = sorted(x for x in glob.glob('data/iwpt2020/train-dev-combined/**/dev.conllu') if 'short' not in x)
    shorten(files)
    files = sorted(x for x in glob.glob('data/iwpt2020/test-udpipe/*.fixed.conllu') if 'short' not in x)
    shorten(files)


def shorten(files):
    pretrained = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    config = AutoConfig.from_pretrained(pretrained)
    for idx, f in enumerate(files):
        langcode = f.split('/')[-2]
        if len(langcode) != 2:
            langcode = os.path.basename(f).split('.')[0]
        # if langcode != 'en':
        #     continue
        print(f'{idx + 1:02d}/{len(files)} {langcode}')
        with open(f.replace('.conllu', '.short.conllu'), 'w') as out:
            long_sent = {}
            oid = 0
            for sid, sent in enumerate(read_conll(f)):
                words = [x[1] for x in sent]
                length = len_of_sent(words, config, tokenizer)
                if length > 256 - 2:
                    assert oid not in long_sent
                    long_sent[oid] = sent_str(sent)
                else:
                    out.write(sent_str(sent))
                    out.write('\n\n')
                oid += 1
            save_json(long_sent, f.replace('.conllu', '.long.json'))


if __name__ == '__main__':
    main()
