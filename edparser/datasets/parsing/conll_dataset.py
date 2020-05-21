# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 16:10
from copy import copy

import torch

from edparser.common.constant import ROOT, CLS
from edparser.common.dataset import TransformDataset

from edparser.components.parsers.alg import kmeans
from torch.utils.data import Sampler
from edparser.components.parsers.conll import read_conll


class CoNLLDataset(TransformDataset):

    def load_file(self, filepath):
        field_names = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                       'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
        for sent in read_conll(filepath):
            sample = {}
            for i, field in enumerate(field_names):
                sample[field] = [cell[i] for cell in sent]
            yield sample

    def __getitem__(self, index: int) -> dict:
        sample = self.data[index]
        sample = copy(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.data)


def append_bos(sample: dict) -> dict:
    sample['token'] = [ROOT] + sample['FORM']
    sample['pos'] = [ROOT] + sample['CPOS']
    sample['arc'] = [0] + sample['HEAD']
    sample['rel'] = sample['DEPREL'][:1] + sample['DEPREL']
    return sample


class BucketSampler(Sampler):
    # noinspection PyMissingConstructor
    def __init__(self, buckets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[
            (size, bucket) for size, bucket in buckets.items()
        ])
        # the number of chunks in each bucket, which is clipped by
        # range [1, len(bucket)]
        self.chunks = [
            min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
            for size, bucket in zip(self.sizes, self.buckets)
        ]

    def __iter__(self):
        # if shuffle, shuffle both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                           for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        return sum(self.chunks)


class KMeansSampler(BucketSampler):
    def __init__(self, lengths, batch_size, shuffle=False, n_buckets=1):
        self.n_buckets = n_buckets
        self.lengths = lengths
        buckets = dict(zip(*kmeans(self.lengths, n_buckets)))
        super().__init__(buckets, batch_size, shuffle)
