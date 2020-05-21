# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-07 23:48
from edparser.components.parsers.conll import CoNLLSentence
from edparser.metrics.parsing.conllx_eval import copy_cols

from edparser.metrics.parsing import conllx_eval
import os
from edparser.components.parsers.biaffine_parser import BiaffineTransformerDependencyParser
from iwpt2020 import cdroot

cdroot()
save_dir = 'data/model/dep/kr_revised_bert2'
input_file = 'data/parsing/ko_penn-ud-revised/ko_penn-ud-revised.conllu'
train_file = input_file.replace('.conllu', '.train.conllu')
dev_file = input_file.replace('.conllu', '.dev.conllu')
test_file = input_file.replace('.conllu', '.test.conllu')
parser = BiaffineTransformerDependencyParser()
# parser.fit(train_file,
#            dev_file,
#            save_dir,
#            'bert-base-multilingual-uncased',
#            batch_size=4000,
#            warmup_steps_ratio=.1,
#            token_mapping=None,
#            samples_per_batch=150,
#            transformer_dropout=.33,
#            learning_rate=5e-05,
#            learning_rate_transformer=5e-06,
#            # max_samples_per_batch=4,
#            # early_stopping_patience=10,
#            )
parser.load(save_dir, tree='tarjan')
parser.config.tree = 'mst'
output = f'{save_dir}/{os.path.basename(test_file).replace(".conllu", ".pred.conllu")}'
parser.evaluate(test_file, save_dir, warm_up=False, output=output)
uas, las = conllx_eval.evaluate(test_file, output)
print(f'Official UAS: {uas:.4f} LAS: {las:.4f}')
print(f'Model saved in {save_dir}')
test_output = f'{save_dir}/{os.path.basename(test_file).replace(".conllu", ".pred.conllu")}'
sents = CoNLLSentence.from_file(test_output)
test_output = test_output.replace(".conllu", ".no2nd.conllu")
with open(test_output, 'w') as out:
    for each in sents:
        for word in each:
            word.phead = None
        out.write(str(each))
        out.write('\n\n')
final_output = test_output.replace('.conllu', '.with_comments.conllu')
copy_cols(test_file, test_output, final_output)
