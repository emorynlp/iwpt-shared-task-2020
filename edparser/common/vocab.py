# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 22:42
from typing import List, Dict, Union, Iterable

from edparser.common.constant import UNK, PAD
from edparser.common.structure import Serializable


class Vocab(Serializable):
    def __init__(self, idx_to_token: List[str] = None, token_to_idx: Dict = None, mutable=True, pad_token=PAD,
                 unk_token=UNK, lowercase=False) -> None:
        super().__init__()
        if idx_to_token:
            t2i = dict((token, idx) for idx, token in enumerate(idx_to_token))
            if token_to_idx:
                t2i.update(token_to_idx)
            token_to_idx = t2i
        if token_to_idx is None:
            token_to_idx = {}
            if pad_token:
                token_to_idx[pad_token] = len(token_to_idx)
            if unk_token:
                token_to_idx[unk_token] = len(token_to_idx)
        self.token_to_idx = token_to_idx
        self.idx_to_token: list = None
        self.mutable = mutable
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.lowercase = lowercase

    def __setitem__(self, token: str, idx: int):
        assert self.mutable, 'Update an immutable Vocab object is not allowed'
        self.token_to_idx[token] = idx

    def __getitem__(self, key: Union[str, int, List]) -> Union[int, str, List]:
        if isinstance(key, str):
            return self.get_idx(key)
        elif isinstance(key, int):
            return self.get_token(key)
        elif isinstance(key, list):
            if len(key) == 0:
                return []
            elif isinstance(key[0], str):
                return [self.get_idx(x) for x in key]
            elif isinstance(key[0], int):
                return [self.get_token(x) for x in key]

    def __contains__(self, key: Union[str, int]):
        if isinstance(key, str):
            return key in self.token_to_idx
        elif isinstance(key, int):
            return 0 <= key < len(self.idx_to_token)
        else:
            return False

    def add(self, token: str) -> int:
        assert self.mutable, 'It is not allowed to call add on an immutable Vocab'
        assert isinstance(token, str), f'Token type must be str but got {type(token)} from {token}'
        assert token, 'Token must not be None or length 0'
        idx = self.token_to_idx.get(token, None)
        if idx is None:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
        return idx

    def update(self, tokens: Iterable[str]) -> None:
        """
        Update the vocab with these tokens by adding them to vocab one by one.
        Parameters
        ----------
        tokens
        """
        assert self.mutable, 'It is not allowed to update an immutable Vocab'
        for token in tokens:
            self.add(token)

    def get_idx(self, token: str) -> int:
        idx = self.token_to_idx.get(token, None)
        if self.lowercase and idx is None:
            idx = self.token_to_idx.get(token.lower(), None)
        if idx is None:
            if self.mutable:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
            else:
                idx = self.token_to_idx.get(self.unk_token, None)
        return idx

    def get_idx_without_add(self, token: str) -> int:
        idx = self.token_to_idx.get(token, None)
        if idx is None:
            idx = self.token_to_idx.get(self.safe_unk_token, None)
        return idx

    def get_token(self, idx: int) -> str:
        if self.idx_to_token:
            return self.idx_to_token[idx]

        if self.mutable:
            for token in self.token_to_idx:
                if self.token_to_idx[token] == idx:
                    return token

    def has_key(self, token):
        return token in self.token_to_idx

    def __len__(self):
        return len(self.token_to_idx)

    def lock(self):
        if self.locked:
            return self
        self.mutable = False
        self.build_idx_to_token()
        return self

    def build_idx_to_token(self):
        max_idx = max(self.token_to_idx.values())
        self.idx_to_token = [None] * (max_idx + 1)
        for token, idx in self.token_to_idx.items():
            self.idx_to_token[idx] = token

    def unlock(self):
        if not self.locked:
            return
        self.mutable = True
        self.idx_to_token = None
        return self

    @property
    def locked(self):
        return not self.mutable

    @property
    def unk_idx(self):
        if self.unk_token is None:
            return None
        else:
            return self.token_to_idx.get(self.unk_token, None)

    @property
    def pad_idx(self):
        if self.pad_token is None:
            return None
        else:
            return self.token_to_idx.get(self.pad_token, None)

    @property
    def tokens(self):
        return self.token_to_idx.keys()

    def __str__(self) -> str:
        return self.token_to_idx.__str__()

    def summary(self, verbose=True) -> str:
        # report = 'Length: {}\n'.format(len(self))
        # report += 'Samples: {}\n'.format(str(list(self.token_to_idx.keys())[:min(50, len(self))]))
        # report += 'Mutable: {}'.format(self.mutable)
        # report = report.strip()
        report = '[{}] = '.format(len(self))
        report += str(list(self.token_to_idx.keys())[:min(50, len(self))])
        if verbose:
            print(report)
        return report

    def __call__(self, some_token: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        if isinstance(some_token, (list, tuple, set)):
            indices = []
            if len(some_token) and isinstance(some_token[0], (list, tuple, set)):
                for sent in some_token:
                    inside = []
                    for token in sent:
                        inside.append(self.get_idx(token))
                    indices.append(inside)
                return indices
            for token in some_token:
                indices.append(self.get_idx(token))
            return indices
        else:
            return self.get_idx(some_token)

    def to_dict(self) -> dict:
        idx_to_token = self.idx_to_token
        pad_token = self.pad_token
        unk_token = self.unk_token
        mutable = self.mutable
        items = locals().copy()
        items.pop('self')
        return items

    def copy_from(self, item: dict):
        for key, value in item.items():
            setattr(self, key, value)
        self.token_to_idx = {k: v for v, k in enumerate(self.idx_to_token)}

    def lower(self):
        self.unlock()
        token_to_idx = self.token_to_idx
        self.token_to_idx = {}
        for token in token_to_idx.keys():
            self.add(token.lower())
        return self

    @property
    def first_token(self):
        if self.idx_to_token:
            return self.idx_to_token[0]
        if self.token_to_idx:
            return next(iter(self.token_to_idx))
        return None

    def merge(self, other):
        for word, idx in other.token_to_idx.items():
            self.get_idx(word)

    @property
    def safe_pad_token(self) -> str:
        """
        Get the pad token safely. It always returns a pad token, which is the token
        closest to pad if not presented in the vocab.

        Returns
        -------
            str pad token
        """
        if self.pad_token:
            return self.pad_token
        if self.first_token:
            return self.first_token
        return PAD

    @property
    def safe_pad_token_idx(self) -> int:
        return self.token_to_idx.get(self.safe_pad_token, 0)

    @property
    def safe_unk_token(self) -> str:
        """
        Get the unk token safely. It always returns a unk token, which is the token
        closest to unk if not presented in the vocab.

        Returns
        -------
            str pad token
        """
        if self.unk_token:
            return self.unk_token
        if self.first_token:
            return self.first_token
        return UNK

    def __repr__(self) -> str:
        if self.idx_to_token is not None:
            return self.idx_to_token.__repr__()
        return self.token_to_idx.__repr__()

    def extend(self, tokens: Iterable[str]):
        self.unlock()
        self(tokens)

    def reload_idx_to_token(self, idx_to_token: List[str], pad_idx=0, unk_idx=1):
        self.idx_to_token = idx_to_token
        self.token_to_idx = dict((s, i) for i, s in enumerate(idx_to_token))
        if pad_idx is not None:
            self.pad_token = idx_to_token[pad_idx]
        if unk_idx is not None:
            self.unk_token = idx_to_token[unk_idx]


def create_label_vocab() -> Vocab:
    return Vocab(pad_token=None, unk_token=None)
