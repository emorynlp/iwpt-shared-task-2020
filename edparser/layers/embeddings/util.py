# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 15:45
import numpy as np
import torch
from torch import nn
from edparser.common.vocab import Vocab
from edparser.utils.io_util import load_word2vec


def index_word2vec_with_vocab(filepath: str, vocab: Vocab, extend_vocab=True, unk=None, lowercase=False, init_unk=nn.init.normal_) -> torch.Tensor:
    word2vec, dim = load_word2vec(filepath)
    if extend_vocab:
        vocab.unlock()
        for word in word2vec:
            vocab.get_idx(word.lower() if lowercase else word)
    vocab.lock()
    if unk and unk in word2vec:
        word2vec[vocab.safe_unk_token] = word2vec.pop(unk)
    pret_embs = torch.zeros(len(vocab), dim)
    for word, idx in vocab.token_to_idx.items():
        vec = word2vec.get(word, None)
        # Retry lower case
        if vec is None:
            vec = word2vec.get(word.lower(), None)
        if vec is not None:
            pret_embs[idx] = torch.tensor(vec)
        else:
            init_unk(pret_embs[idx])
    return pret_embs


def build_word2vec_with_vocab(embed, vocab: Vocab, extend_vocab=True, unk=None, lowercase=False) -> nn.Embedding:
    if isinstance(embed, str):
        embed = index_word2vec_with_vocab(embed, vocab, extend_vocab, unk, lowercase)
        embed = nn.Embedding.from_pretrained(embed, False, padding_idx=vocab.pad_idx)
        return embed
    elif isinstance(embed, int):
        embed = nn.Embedding(len(vocab), embed, padding_idx=vocab.pad_idx)
        return embed
    else:
        raise ValueError(f'Unsupported parameter type: {embed}')
