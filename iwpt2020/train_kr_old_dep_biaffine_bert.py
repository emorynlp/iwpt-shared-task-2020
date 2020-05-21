# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-07 23:48

from edparser.components.parsers.biaffine_parser import BiaffineTransformerDependencyParser
from iwpt2020 import cdroot

cdroot()
save_dir = 'data/model/dep/kr_old_bert3'
parser = BiaffineTransformerDependencyParser()
parser.fit('data/parsing/ud_ko_old/ko_penn-ud.train.conllu',
           'data/parsing/ud_ko_old/ko_penn-ud.dev.conllu', save_dir, 'bert-base-multilingual-uncased',
           batch_size=1000,
           warmup_steps_ratio=.1,
           token_mapping=None,
           samples_per_batch=150,
           transformer_dropout=.33,
           learning_rate=5e-05,
           learning_rate_transformer=5e-06,
           # early_stopping_patience=10,
           )
parser.config.tree = 'tarjan'
# parser.load(save_dir, tree='tarjan')
output = f'{save_dir}/ko_penn-ud.test.predict.conll'
test = 'data/parsing/ud_ko_old/ko_penn-ud.test.conllu'
parser.evaluate(test, save_dir, warm_up=False, output=output)
# uas, las = conllx_eval.evaluate(test, output)
# print(f'Official UAS: {uas:.4f} LAS: {las:.4f}')
print(f'Model saved in {save_dir}')
