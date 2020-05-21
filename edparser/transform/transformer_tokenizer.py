# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-03 16:23
import warnings
from typing import Union

from edparser.layers.transformers.pt_imports import PreTrainedTokenizer, PretrainedConfig, AutoTokenizer


class TransformerTextTokenizer(object):
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, str],
                 input_key,
                 output_key=None,
                 max_seq_length=512,
                 config: PretrainedConfig = None,
                 cls_token_at_end=False,
                 cls_token_segment_id=0,
                 pad_token_segment_id=0,
                 pad_on_left=False,
                 do_padding=False,
                 sep_token_extra=False) -> None:
        self.input_key = input_key
        if not output_key:
            output_key = [f'{input_key}_{key}' for key in ['input_ids', 'attention_mask', 'token_type_ids']]
        self.output_key = output_key
        if config:
            xlnet = config_is(config, 'xlnet')
            roberta = config_is(config, 'roberta')
            pad_token_segment_id = 4 if xlnet else 0
            cls_token_segment_id = 2 if xlnet else 0
            sep_token_extra = roberta
            cls_token_at_end = xlnet
            pad_on_left = xlnet
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.pad_token_segment_id = pad_token_segment_id
        self.cls_token_segment_id = cls_token_segment_id
        self.sep_token_extra = sep_token_extra
        self.cls_token_at_end = cls_token_at_end
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.pad_on_left = pad_on_left
        self.do_padding = do_padding

    def __call__(self, sample: dict):
        input_ids, attention_mask, token_type_ids, prefix_mask = \
            convert_examples_to_features([sample[self.input_key]],
                                         self.max_seq_length,
                                         self.tokenizer,
                                         cls_token_at_end=self.cls_token_at_end,
                                         # xlnet has a cls token at the end
                                         cls_token=self.tokenizer.cls_token,
                                         cls_token_segment_id=self.cls_token_segment_id,
                                         sep_token=self.sep_token,
                                         sep_token_extra=self.sep_token_extra,
                                         # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                         pad_on_left=self.pad_on_left,
                                         # pad on the left for xlnet
                                         pad_token_id=self.pad_token_id,
                                         pad_token_segment_id=self.pad_token_segment_id,
                                         pad_token_label_id=0,
                                         do_padding=self.do_padding)
        for k, v in zip(self.output_key, [input_ids, attention_mask, token_type_ids]):
            sample[k] = v
        return sample


class TransformerSequenceTokenizer(TransformerTextTokenizer):

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, str], input_key, output_key=None, max_seq_length=512,
                 config: PretrainedConfig = None, cls_token_at_end=False, cls_token_segment_id=0,
                 pad_token_segment_id=0, pad_on_left=False, do_padding=False, sep_token_extra=False,
                 ret_token_span=False, cls_is_bos=False) -> None:
        self.cls_is_bos = cls_is_bos
        self.ret_token_span = ret_token_span
        if not output_key:
            suffixes = ['input_ids', 'attention_mask', 'token_type_ids', 'prefix_mask']
            if ret_token_span:
                suffixes.append('token_span')
            output_key = [f'{input_key}_{key}' for key in suffixes]
        super().__init__(tokenizer, input_key, output_key, max_seq_length, config, cls_token_at_end,
                         cls_token_segment_id, pad_token_segment_id, pad_on_left, do_padding, sep_token_extra)
        if self.ret_token_span:
            assert not self.cls_token_at_end
            assert not self.pad_on_left

    def __call__(self, sample: dict):
        input_tokens = sample[self.input_key]
        if self.cls_is_bos:
            input_tokens = input_tokens[1:]
        input_ids, attention_mask, token_type_ids, prefix_mask = \
            convert_examples_to_features(input_tokens,
                                         self.max_seq_length,
                                         self.tokenizer,
                                         cls_token_at_end=self.cls_token_at_end,
                                         # xlnet has a cls token at the end
                                         cls_token=self.tokenizer.cls_token,
                                         cls_token_segment_id=self.cls_token_segment_id,
                                         sep_token=self.sep_token,
                                         sep_token_extra=self.sep_token_extra,
                                         # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                         pad_on_left=self.pad_on_left,
                                         # pad on the left for xlnet
                                         pad_token_id=self.pad_token_id,
                                         pad_token_segment_id=self.pad_token_segment_id,
                                         pad_token_label_id=0,
                                         do_padding=self.do_padding)
        if self.cls_is_bos:
            prefix_mask[0] = True
        outputs = [input_ids, attention_mask, token_type_ids, prefix_mask]
        if self.ret_token_span:
            if self.cls_is_bos:
                token_span = [[0]]
            else:
                token_span = []
            offset = 1
            span = []
            for mask in prefix_mask[1:-1]:  # skip [CLS] and [SEP]
                if mask and span:
                    token_span.append(span)
                    span = []
                span.append(offset)
                offset += 1
            if span:
                token_span.append(span)
            outputs.append(token_span)
        for k, v in zip(self.output_key, outputs):
            sample[k] = v
        return sample


def config_is(config, model='bert'):
    return model in type(config).__name__.lower()


def convert_examples_to_features(
        words,
        max_seq_length,
        tokenizer,
        labels=None,
        label_map=None,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token_id=0,
        pad_token_segment_id=0,
        pad_token_label_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        unk_token='[UNK]',
        do_padding=True
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    args = locals()
    if not labels:
        labels = words
        pad_token_label_id = False

    tokens = []
    label_ids = []
    for word, label in zip(words, labels):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            # some wired chars cause the tagger to return empty list
            word_tokens = [unk_token] * len(word)
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        label_ids.extend([label_map[label] if label_map else True] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        warnings.warn(
            f'Input tokens {words} exceed the max sequence length of {max_seq_length - special_tokens_count}. '
            f'The exceeded part will be truncated and ignored. '
            f'You are recommended to split your long text into several sentences within '
            f'{max_seq_length - special_tokens_count} tokens beforehand.')
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    if do_padding:
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token_id] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token_id] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length, f'failed for:\n {args}'
    else:
        assert len(set(len(x) for x in [input_ids, input_mask, segment_ids, label_ids])) == 1
    return input_ids, input_mask, segment_ids, label_ids


def main():
    transformer = 'bert-base-uncased'
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer)
    # _test_text_transform(tokenizer)
    _test_sequence_transform(tokenizer)


def _test_text_transform(tokenizer):
    transform = TransformerTextTokenizer(tokenizer, 'text')
    sample = {'text': 'HanLP good'}
    print(transform(sample))


def _test_sequence_transform(tokenizer):
    transform = TransformerSequenceTokenizer(tokenizer, 'text', ret_token_span=True)
    sample = {'text': 'HanLP good'.split()}
    print(transform(sample))


if __name__ == '__main__':
    main()
