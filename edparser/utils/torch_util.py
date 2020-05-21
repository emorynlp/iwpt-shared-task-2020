# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 15:52
import os
from typing import List

import random
import torch
from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit, nvmlShutdown
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from edparser.utils.log_util import logger


def gpus_available() -> dict:
    try:
        nvmlInit()
        gpus = {}
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_devices:
            visible_devices = {int(x.strip()) for x in visible_devices.split(',')}
        else:
            visible_devices = list(range(nvmlDeviceGetCount()))
        for i, real_id in enumerate(visible_devices):
            h = nvmlDeviceGetHandleByIndex(real_id)
            info = nvmlDeviceGetMemoryInfo(h)
            total = info.total
            free = info.free
            ratio = free / total
            gpus[i] = ratio
            # print(f'total    : {info.total}')
            # print(f'free     : {info.free}')
            # print(f'used     : {info.used}')
            # t = torch.cuda.get_device_properties(0).total_memory
            # c = torch.cuda.memory_cached(0)
            # a = torch.cuda.memory_allocated(0)
            # print(t, c, a)
        nvmlShutdown()
        return gpus
    except Exception as e:
        logger.debug(f'Failed to get gpu info due to {e}')
        return {}


def cuda_devices(query=None) -> List[int]:
    """
    Decide which GPUs to use

    Parameters
    ----------
    query

    Returns
    -------

    """
    if isinstance(query, list):
        return query
    if query is None:
        query = gpus_available()
        if not query:
            return []
        query = max((v, k) for k, v in query.items())[-1]
    if isinstance(query, float):
        gpus = gpus_available()
        if not query:
            return []
        query = [k for k, v in gpus.items() if v > query]
    elif isinstance(query, int):
        query = [query]
    return query


def pad_lists(sequences: List[List], dtype=torch.long, padding_value=0):
    return pad_sequence([torch.tensor(x, dtype=dtype) for x in sequences], True, padding_value)


def set_seed(seed):
    """
    Copied from https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L76
    Parameters
    ----------
    seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


def main():
    print(gpus_available())
    print(cuda_devices())
    print(cuda_devices(0.1))


if __name__ == '__main__':
    main()
