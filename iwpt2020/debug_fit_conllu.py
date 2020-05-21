# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-27 00:35

from edparser.layers.transformers import PreTrainedTokenizer, TFAutoModel, TFPreTrainedModel, AutoTokenizer

from edparser.components.parsers.conll import CoNLL_Transformer_Transform
from iwpt2020 import cdroot

transformer = 'bert-base-uncased'
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer)
transformer: TFPreTrainedModel = TFAutoModel.from_pretrained(transformer, name=transformer)

transform = CoNLL_Transformer_Transform(graph=True)
transform.transformer_config = transformer.config
transform.tokenizer = tokenizer

cdroot()
valid = 'data/iwpt2020/train-dev/UD_English-EWT/en_ewt-ud-dev.enhanced_collapse_empty_nodes.conllu'
transform.fit(valid)
transform.lock_vocabs()
transform.summarize_vocabs()
dataset = transform.file_to_dataset(valid)
batch = next(iter(dataset))
print(batch)
print(sum(len(x[0][0]) for x in iter(dataset)))
