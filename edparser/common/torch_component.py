# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 21:20
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from edparser.common.component import Component
from edparser.common.structure import SerializableDict
from edparser.common.transform import VocabDict
from edparser.utils.io_util import get_resource
from edparser.utils.log_util import logger, init_logger
from edparser.utils.torch_util import cuda_devices
from edparser.utils.util import merge_dict, isdebugging


class TorchComponent(Component, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model: Optional[torch.nn.Module] = None
        self.config = SerializableDict(**kwargs)
        self.vocabs: Optional[VocabDict] = None

    def _capture_config(self, locals_: Dict,
                        exclude=(
                                'trn_data', 'dev_data', 'save_dir', 'kwargs', 'self', 'logger', 'verbose',
                                'dev_batch_size', '__class__', 'devices')):
        """
        Save arguments to config

        Parameters
        ----------
        locals_
            `locals()`
        exclude
        """
        if 'kwargs' in locals_:
            locals_.update(locals_['kwargs'])
        locals_ = dict(locals_.items())
        for key in exclude:
            locals_.pop(key, None)
        self.config.update(locals_)
        return self.config

    def save_weights(self, save_dir, filename='model.pt', **kwargs):
        torch.save(self.model.state_dict(), os.path.join(save_dir, filename))

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        save_dir = get_resource(save_dir)
        self.model.load_state_dict(torch.load(os.path.join(save_dir, filename)), strict=False)

    def save_config(self, save_dir, filename='config.json'):
        self.config.save_json(os.path.join(save_dir, filename))

    def load_config(self, save_dir, filename='config.json'):
        save_dir = get_resource(save_dir)
        self.config.load_json(os.path.join(save_dir, filename))

    def save_vocabs(self, save_dir):
        self.vocabs.save_vocabs(save_dir)

    def load_vocabs(self, save_dir):
        self.vocabs = VocabDict()
        self.vocabs.load_vocabs(save_dir)

    def save(self, save_dir: str, **kwargs):
        self.save_config(save_dir)
        self.save_vocabs(save_dir)
        self.save_weights(save_dir)

    def load(self, save_dir: str, device=None, **kwargs):
        save_dir = get_resource(save_dir)
        self.load_config(save_dir)
        self.load_vocabs(save_dir)
        self.model = self.build_model(
            **merge_dict(self.config, training=False, logger=logger, **kwargs, overwrite=True,
                         inplace=True))
        self.to(device)
        self.load_weights(save_dir, **kwargs)

    def fit(self, trn_data, dev_data, save_dir, batch_size, epochs, devices=None, logger=None, verbose=True, **kwargs):
        # Common initialization steps
        config = self._capture_config(locals())
        if not logger:
            logger = init_logger(name='train', root_dir=save_dir, level=logging.INFO if verbose else logging.WARN)
        devices = cuda_devices(devices)
        trn = self.build_dataloader(trn_data, batch_size, True, devices[0], logger)
        dev = self.build_dataloader(dev_data, batch_size, False, devices[0], logger)
        self.save_config(save_dir)
        self.save_vocabs(save_dir)
        self.model = self.build_model(**config)
        self.to(devices, logger)
        criterion = self.build_criterion(**self.config)
        optimizer = self.build_optimizer(**self.config)
        metric = self.build_metric(**self.config)
        return self.run_fit(**merge_dict(config, trn=trn, dev=dev, epochs=epochs, criterion=criterion,
                                         optimizer=optimizer, metric=metric, logger=logger, save_dir=save_dir,
                                         overwrite=True))

    @abstractmethod
    def build_optimizer(self, **kwargs):
        pass

    @abstractmethod
    def build_criterion(self, **kwargs):
        pass

    @abstractmethod
    def build_metric(self, **kwargs):
        pass

    @abstractmethod
    def run_fit(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir, logger,
                **kwargs):
        pass

    @abstractmethod
    def fit_dataloader(self, trn: DataLoader, optimizer, **kwargs):
        pass

    @abstractmethod
    def evaluate_dataloader(self, data: DataLoader, **kwargs):
        pass

    @abstractmethod
    def build_model(self, **kwargs) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_dataloader(self, data, batch_size, shuffle, device, logger, **kwargs) -> DataLoader:
        pass

    def evaluate(self, tst_path, batch_size=None, save_dir=None, logger=None, **kwargs):
        if logger is None:
            logger = init_logger(name='train', root_dir=save_dir, level=logging.INFO)
        if not batch_size:
            batch_size = self.config.get('batch_size', 32)
        dataset = self.build_dataloader(tst_path, batch_size, False, self.devices[0], **kwargs)
        return self.evaluate_dataloader(data=dataset,
                                        **merge_dict(self.config, batch_size=batch_size, logger=logger, **kwargs))

    def to(self, devices=Union[str, List[int]], logger=None):
        if isinstance(devices, str):
            devices = cuda_devices(devices)
        if devices:
            if logger:
                logger.info(f'Using GPUs: {devices}')
            self.model = self.model.to(devices[0])
            if len(devices) > 1 and not isdebugging():
                self.model = nn.DataParallel(self.model, device_ids=devices)
        else:
            if logger:
                logger.info('Using CPU')

    @property
    def devices(self):
        if self.model is None:
            return None
        # next(parser.model.parameters()).device
        if hasattr(self.model, 'device_ids'):
            return self.model.device_ids
        device: torch.device = next(self.model.parameters()).device
        return [device]
