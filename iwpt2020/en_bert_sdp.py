# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-26 18:19
from edparser.components.parsers.biaffine_parser import BiaffineTransformerSemanticDependencyParser
from iwpt2020 import cdroot

cdroot()
save_dir = 'data/model/iwpt2020/en/bert_sdp'
parser = BiaffineTransformerSemanticDependencyParser()
dataset = 'data/iwpt2020/train-dev/'
trnfile = f'{dataset}UD_English-EWT/en_ewt-ud-train.enhanced_collapse_empty_nodes.conllu'
devfile = f'{dataset}UD_English-EWT/en_ewt-ud-dev.enhanced_collapse_empty_nodes.conllu'
testfile = devfile
# parser.fit(trnfile,
#            devfile,
#            save_dir,
#            'bert-base-uncased',
#            batch_size=5000,
#            warmup_steps_ratio=.1,
#            samples_per_batch=150,
#            # max_samples_per_batch=80,
#            transformer_dropout=.33,
#            learning_rate=2e-3,
#            learning_rate_transformer=1e-5,
#            # epochs=1
#            )
parser.load(save_dir)
parser.evaluate('data/iwpt2020/train-dev/UD_English-EWT/en_ewt-ud-test.conllu')
# output = f'{testfile.replace(".conllu", ".pred.conllu")}'
# output = f'{save_dir}/{os.path.basename(output)}'
# logger = init_logger(name='test', root_dir=save_dir, mode='w')
# parser.evaluate(testfile, save_dir, warm_up=False, output=output, logger=logger)
# score = evaluate(testfile, output)
# logger.info(f'ELAS: {score["ELAS"].f1 * 100:.2f} - CLAS:{score["CLAS"].f1 * 100:.2f}')
# print(f'Model saved in {save_dir}')
