# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-26 18:19
from edparser.components.parsers.conll import CoNLLSentence, CoNLLUWord
from edparser.components.parsers.parse_alg import adjust_root_score_then_add_secondary_arcs
from edparser.metrics.parsing.iwpt20_eval import evaluate
from edparser.components.parsers.biaffine_parser import BiaffineTransformerSemanticDependencyParser
from edparser.utils.io_util import save_pickle, load_pickle
from edparser.utils.log_util import init_logger
from iwpt2020 import cdroot
import os

cdroot()
save_dir = 'data/model/iwpt2020/en/albert_sdp'
parser = BiaffineTransformerSemanticDependencyParser()
dataset = 'data/iwpt2020/train-dev/'
trnfile = f'{dataset}UD_English-EWT/en_ewt-ud-train.enhanced_collapse_empty_nodes.conllu'
devfile = f'{dataset}UD_English-EWT/en_ewt-ud-dev.enhanced_collapse_empty_nodes.conllu'
testfile = devfile
# parser.fit(trnfile,
#            devfile,
#            save_dir,
#            'albert-xxlarge-v2',
#            batch_size=1024,
#            warmup_steps_ratio=.1,
#            samples_per_batch=150,
#            max_samples_per_batch=75,
#            transformer_dropout=.33,
#            learning_rate=2e-3,
#            learning_rate_transformer=1e-5,
#            # enhanced_only=True,
#            # epochs=1
#            )
parser.load(save_dir)
output = f'{testfile.replace(".conllu", ".pred.conllu")}'
output = f'{save_dir}/{os.path.basename(output)}'
logger = init_logger(name='test', root_dir=save_dir, mode='w')
pkl_path = f'{save_dir}/sdp.pkl'
try:
    scores = load_pickle(pkl_path)
except FileNotFoundError:
    scores = parser.evaluate(testfile, save_dir, warm_up=False, output=output, ret_scores=True, logger=logger)[-1]
    save_pickle(scores, pkl_path)

with open(output, 'w') as out:
    num = 0
    trees = CoNLLSentence.from_file(
        '/home/hhe43/hanlp/data/model/iwpt2020/en/albert_dep2/en_ewt-ud-dev.enhanced_collapse_empty_nodes.pred.fixed.conllu')
    for arc_scores, rel_scores, mask in scores:
        for a, r, m in zip(arc_scores, rel_scores, mask):
            # tree, graph = mst_then_greedy(a, r, m)
            tree = [0] + [x.head for x in trees[num]]
            graph = adjust_root_score_then_add_secondary_arcs(a, r, tree)
            sent = CoNLLSentence()
            for i, (t, g) in enumerate(zip(tree, graph)):
                if not i:
                    continue
                sent.append(CoNLLUWord(id=i, form=None, head=[x[0] for x in g],
                                       deprel=[parser.transform.rel_vocab.idx_to_token[x[1]] for x in g]))
            out.write(f'{sent}\n\n')
            num += 1
score = evaluate(testfile, output)
logger.info(f'ELAS: {score["ELAS"].f1 * 100:.2f} - CLAS:{score["CLAS"].f1 * 100:.2f}')
print(f'Model saved in {save_dir}')
