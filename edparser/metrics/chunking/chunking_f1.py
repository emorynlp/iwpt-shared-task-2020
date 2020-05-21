# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-16 21:52
from edparser.common.vocab import Vocab
import torch


class ChunkingF1(object):
    def __init__(self, tag_vocab: Vocab):
        self.tag_vocab = tag_vocab
