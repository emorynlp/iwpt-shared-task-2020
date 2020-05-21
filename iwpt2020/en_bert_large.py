# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-26 18:19
from edparser.metrics.parsing.iwpt20_eval import evaluate
from edparser.components.parsers.biaffine_parser import BiaffineTransformerDependencyParser
from iwpt2020 import cdroot

cdroot()
save_dir = 'data/model/iwpt2020/en_bert_large_dep'
parser = BiaffineTransformerDependencyParser()
dataset = 'data/iwpt2020/train-dev/'
trnfile = f'{dataset}UD_English-EWT/en_ewt-ud-train.enhanced_collapse_empty_nodes.conllu'
devfile = f'{dataset}UD_English-EWT/en_ewt-ud-dev.enhanced_collapse_empty_nodes.conllu'
testfile = devfile
# parser.fit(trnfile,
#            devfile,
#            save_dir, 'bert-large-uncased-whole-word-masking',
#            batch_size=128,
#            warmup_steps_ratio=.1,
#            samples_per_batch=150,
#            # max_samples_per_batch=32,
#            transformer_dropout=.33,
#            learning_rate=2e-3,
#            learning_rate_transformer=1e-5,
#            epochs=1
#            )
# parser.load(save_dir, tree='tarjan')
output = f'{testfile.replace(".conllu", ".pred.conllu")}'
# parser.evaluate(devfile, save_dir, warm_up=False, output=output)
score = evaluate(testfile, output)
print(f'ELAS: {score["ELAS"].f1 * 100:.2f} - CLAS:{score["CLAS"].f1 * 100:.2f}')
print(f'Model saved in {save_dir}')
