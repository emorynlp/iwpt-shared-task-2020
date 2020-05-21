# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 20:27
from abc import ABC, abstractmethod
from copy import copy
from typing import Union, List, Callable, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from edparser.common.transform import TransformList, VocabDict
from edparser.common.vocab import Vocab
from edparser.utils.util import isdebugging, merge_list_of_dict


class TransformDataset(Dataset, ABC):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None) -> None:
        super().__init__()
        if isinstance(data, str):
            data = list(self.load_file(data))
        self.data = data
        if isinstance(transform, list) and not isinstance(transform, TransformList):
            transform = TransformList(*transform)
        self.transform: Union[Callable, TransformList] = transform

    @abstractmethod
    def load_file(self, filepath):
        pass

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            assert len(index) == 1
            index = index[0]
        sample = self.data[index]
        sample = copy(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.data)


class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=None, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 device=None, **kwargs):
        if batch_sampler is not None:
            batch_size = 1
        if num_workers is None:
            if isdebugging():
                num_workers = 0
            else:
                num_workers = 2
        # noinspection PyArgumentList
        super(DeviceDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                               sampler=sampler,
                                               batch_sampler=batch_sampler, num_workers=num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                               worker_init_fn=worker_init_fn,
                                               multiprocessing_context=multiprocessing_context, **kwargs)
        self.device = device

    def __iter__(self):
        for raw_batch in super(DeviceDataLoader, self).__iter__():
            if self.device is not None:
                for field, data in raw_batch.items():
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device)
                        raw_batch[field] = data
            yield raw_batch

    def collate_fn(self, samples):
        return merge_list_of_dict(samples)


class PadSequenceDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=32, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 pad: dict = None, vocabs: VocabDict = None, device=None, **kwargs):
        if collate_fn is None:
            collate_fn = self.collate_fn
        if num_workers is None:
            if isdebugging():
                num_workers = 0
            else:
                num_workers = 2
        if batch_sampler is not None:
            batch_size = 1
        # noinspection PyArgumentList
        super(PadSequenceDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                    sampler=sampler,
                                                    batch_sampler=batch_sampler, num_workers=num_workers,
                                                    collate_fn=collate_fn,
                                                    pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                                    worker_init_fn=worker_init_fn,
                                                    multiprocessing_context=multiprocessing_context, **kwargs)
        self.vocabs = vocabs
        self.pad = pad
        self.device = device

    def __iter__(self):
        for raw_batch in super(PadSequenceDataLoader, self).__iter__():
            for field, data in raw_batch.items():
                vocab_key = field[:-len('_id')] if field.endswith('_id') else None
                vocab: Vocab = self.vocabs.get(vocab_key, None) if self.vocabs and vocab_key else None
                if vocab:
                    pad = vocab.safe_pad_token_idx
                elif self.pad is not None and field in self.pad:
                    pad = self.pad[field]
                else:
                    # no need to pad
                    continue
                if isinstance(data[0], torch.Tensor):
                    data = pad_sequence(data, True, pad)
                elif isinstance(data[0], Iterable):
                    if isinstance(data[0][0], Iterable):
                        max_seq_len = len(max(data, key=len))
                        max_word_len = len(max([chars for words in data for chars in words], key=len))
                        ids = torch.zeros(len(data), max_seq_len, max_word_len, dtype=torch.long)
                        for i, words in enumerate(data):
                            for j, chars in enumerate(words):
                                ids[i][j][:len(chars)] = torch.tensor(chars)
                        data = ids
                    else:
                        data = pad_sequence([torch.tensor(x) for x in data], True, pad)
                raw_batch[field] = data
            if self.device is not None:
                for field, data in raw_batch.items():
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device)
                        raw_batch[field] = data
            yield raw_batch

    def collate_fn(self, samples):
        return merge_list_of_dict(samples)
