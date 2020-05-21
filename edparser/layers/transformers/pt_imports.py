# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 11:25
import os
from typing import Tuple

import torch

if os.environ.get('USE_TF', None) is None:
    os.environ["USE_TF"] = 'NO'  # saves time loading transformers
from transformers import BertTokenizer, BertConfig, PretrainedConfig, \
    AutoConfig, AutoTokenizer, PreTrainedTokenizer, BertTokenizerFast, AlbertConfig, BertModel, AutoModel, \
    PreTrainedModel, get_linear_schedule_with_warmup, AdamW


def get_optimizers(
        model: torch.nn.Module,
        num_training_steps: int,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        warmup_steps=0.1,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Modified from https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L232
    Setup the optimizer and the learning rate scheduler.

    We provide a reasonable default that works well.
    """
    if isinstance(warmup_steps, float):
        assert 0 < warmup_steps < 1
        warmup_steps = int(num_training_steps * warmup_steps)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler
