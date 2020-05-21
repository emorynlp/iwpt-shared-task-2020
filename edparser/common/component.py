# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-26 14:45
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict


from edparser.version import __version__
from edparser.utils.reflection import class_path_of, str_to_type


class Component(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.meta = {
            'class_path': class_path_of(self),
            'hanlp_version': __version__,
        }

    @abstractmethod
    def predict(self, data: Any, **kwargs):
        """
        Predict on data
        :param data: Any type of data subject to sub-classes
        :param kwargs: Additional arguments
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def __call__(self, data, **kwargs):
        return self.predict(data, **kwargs)

    @staticmethod
    def from_meta(meta: dict, **kwargs):
        """

        Parameters
        ----------
        meta
        kwargs

        Returns
        -------
        Component
        """
        cls = meta.get('class_path', None)
        assert cls, f'{meta} doesn\'t contain class_path field'
        cls = str_to_type(cls)
        return cls.from_meta(meta)