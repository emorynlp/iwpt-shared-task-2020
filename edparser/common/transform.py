# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-03 14:44
import os
from abc import ABC, abstractmethod
from typing import Tuple, Union

from torch.utils.data import Dataset

from edparser.common.structure import SerializableDict
from edparser.common.vocab import Vocab
from edparser.utils.io_util import get_resource


class Transform(ABC):

    def __init__(self, config: SerializableDict = None, **kwargs) -> None:
        super().__init__()
        if kwargs:
            if not config:
                config = SerializableDict()
            for k, v in kwargs.items():
                config[k] = v
        self.config = config

    @abstractmethod
    def fit(self, trn_path: str, **kwargs) -> Union[int, Dataset]:
        """
        Build the vocabulary from training file

        Parameters
        ----------
        trn_path : path to training set
        kwargs

        Returns
        -------
        int
            How many samples in the training set
        """
        pass

    @abstractmethod
    def file_to_dataset(self, filepath: str, **kwargs) -> Dataset:
        """
        Transform file to dataset

        Parameters
        ----------
        filepath

        Returns
        -------

        """
        pass

    @abstractmethod
    def inputs_to_dataset(self, inputs) -> Dataset:
        pass


class ToIndex(ABC):

    def __init__(self, vocab: Vocab = None) -> None:
        super().__init__()
        if vocab is None:
            vocab = Vocab()
        self.vocab = vocab

    @abstractmethod
    def __call__(self, sample):
        pass

    def save_vocab(self, save_dir, filename='vocab.json'):
        vocab = SerializableDict()
        vocab.update(self.vocab.to_dict())
        vocab.save_json(os.path.join(save_dir, filename))

    def load_vocab(self, save_dir, filename='vocab.json'):
        save_dir = get_resource(save_dir)
        vocab = SerializableDict()
        vocab.load_json(os.path.join(save_dir, filename))
        self.vocab.copy_from(vocab)


class FieldToIndex(ToIndex):

    def __init__(self, src, dst=None, vocab=None) -> None:
        super().__init__(vocab)
        self.src = src
        if not dst:
            dst = f'{src}_id'
        self.dst = dst

    def __call__(self, sample: dict):

        sample[self.src] = self.vocab(sample[self.src])
        return sample

    def save_vocab(self, save_dir, filename=None):
        if not filename:
            filename = f'{self.dst}_vocab.json'
        super().save_vocab(save_dir, filename)

    def load_vocab(self, save_dir, filename=None):
        if not filename:
            filename = f'{self.dst}_vocab.json'
        super().load_vocab(save_dir, filename)


class VocabList(list):

    def __init__(self, *fields) -> None:
        super().__init__()
        for each in fields:
            self.append(FieldToIndex(each))

    def append(self, item: Union[str, Tuple[str, Vocab], Tuple[str, str, Vocab], FieldToIndex]) -> None:
        if isinstance(item, str):
            item = FieldToIndex(item)
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                item = FieldToIndex(src=item[0], vocab=item[1])
            elif len(item) == 3:
                item = FieldToIndex(src=item[0], dst=item[1], vocab=item[2])
            else:
                raise ValueError(f'Unsupported argument length: {item}')
        elif isinstance(item, FieldToIndex):
            pass
        else:
            raise ValueError(f'Unsupported argument type: {item}')
        super(self).append(item)

    def save_vocab(self, save_dir):
        for each in self:
            each.save_vocab(save_dir, None)

    def load_vocab(self, save_dir):
        for each in self:
            each.load_vocab(save_dir, None)


class VocabDict(dict):

    def __init__(self, *args, **kwargs) -> None:
        vocabs = dict(kwargs)
        for each in args:
            vocabs[each] = Vocab()
        super().__init__(vocabs)

    def save_vocabs(self, save_dir, filename='vocabs.json'):
        vocabs = SerializableDict()
        for key, value in self.items():
            if isinstance(value, Vocab):
                vocabs[key] = value.to_dict()
        vocabs.save_json(os.path.join(save_dir, filename))

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        save_dir = get_resource(save_dir)
        vocabs = SerializableDict()
        vocabs.load_json(os.path.join(save_dir, filename))
        for key, value in vocabs.items():
            vocab = Vocab()
            vocab.copy_from(value)
            self[key] = vocab

    def lock(self):
        for key, value in self.items():
            if isinstance(value, Vocab):
                value.lock()

    def __call__(self, sample: dict):
        for key, value in self.items():
            if isinstance(value, Vocab):
                field = sample.get(key, None)
                if field is not None:
                    sample[f'{key}_id'] = value(field)
        return sample

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __getitem__(self, k: str) -> Vocab:
        return super().__getitem__(k)

    def __setitem__(self, k: str, v: Vocab) -> None:
        super().__setitem__(k, v)

    def summary(self):
        for key, value in self.items():
            if isinstance(value, Vocab):
                report = value.summary(verbose=False)
                print(f'{key}{report}')


class RenameField(object):
    def __init__(self, src, dst) -> None:
        self.dst = dst
        self.src = src

    def __call__(self, sample: dict):
        sample[self.dst] = sample.pop(self.src)
        return sample


class CopyField(object):
    def __init__(self, src, dst) -> None:
        self.dst = dst
        self.src = src

    def __call__(self, sample: dict) -> dict:
        sample[self.dst] = sample[self.src]
        return sample


class FilterField(object):
    def __init__(self, *keys) -> None:
        self.keys = keys

    def __call__(self, sample: dict):
        sample = dict((k, sample[k]) for k in self.keys)
        return sample


class TransformList(list):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.TransformList(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> )
    """

    def __init__(self, *transforms) -> None:
        super().__init__()
        self.extend(transforms)

    def __call__(self, sample):
        for t in self:
            sample = t(sample)
        return sample


class LowerCase(object):
    def __init__(self, src, dst=None) -> None:
        if dst is None:
            dst = src
        self.src = src
        self.dst = dst

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if isinstance(src, str):
            sample[self.dst] = src.lower()
        elif isinstance(src, list):
            sample[self.dst] = [x.lower() for x in src]
        return sample


class ToChar(object):
    def __init__(self, src, dst='char', max_word_length=None) -> None:
        if dst is None:
            dst = src
        self.src = src
        self.dst = dst
        self.max_word_length = max_word_length

    def __call__(self, sample: dict) -> dict:
        src = sample[self.src]
        if isinstance(src, str):
            sample[self.dst] = self.to_chars(src)
        elif isinstance(src, list):
            sample[self.dst] = [self.to_chars(x) for x in src]
        return sample

    def to_chars(self, word: str):
        if not self.max_word_length:
            return list(word)
        return list(word[:self.max_word_length])
