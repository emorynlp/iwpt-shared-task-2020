# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-26 18:19
from edparser.metrics.parsing.iwpt20_eval import evaluate
from edparser.components.parsers.biaffine_parser import BiaffineTransformerSemanticDependencyParser
from iwpt2020 import cdroot

cdroot()
save_dir = 'data/model/iwpt2020/en_albert_large_graph'
parser = BiaffineTransformerSemanticDependencyParser()
dataset = 'data/iwpt2020/train-dev/'
trnfile = f'{dataset}UD_English-EWT/en_ewt-ud-train.enhanced_collapse_empty_nodes.conllu'
devfile = f'{dataset}UD_English-EWT/en_ewt-ud-dev.enhanced_collapse_empty_nodes.conllu'
trnfile = devfile
testfile = devfile
# parser.fit(trnfile,
#            devfile,
#            save_dir,
#            'albert-xxlarge-v2',
#            batch_size=128,
#            warmup_steps_ratio=.1,
#            samples_per_batch=150,
#            max_samples_per_batch=8,
#            transformer_dropout=.33,
#            learning_rate=2e-3,
#            learning_rate_transformer=1e-5,
#            epochs=1
#            )
parser.load(save_dir)
output = f'{testfile.replace(".conllu", ".pred.conllu")}'
parser.evaluate(devfile, save_dir, warm_up=False, output=output)
score = evaluate(testfile, output)
print(f'ELAS: {score["ELAS"] * 100:.2f} - CLAS:{score["CLAS"] * 100:.2f}')
print(f'Model saved in {save_dir}')
