# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-09 07:45

import glob
import os
from shutil import copyfile
import numpy as np
from edparser.components.parsers.parse_alg import adjust_root_score_then_add_secondary_arcs, mst_then_greedy

from edparser.components.parsers.conll import CoNLLSentence, CoNLLUWord
from edparser.metrics.parsing.iwpt20_eval import evaluate, remove_complete_edges, restore_collapse_edges, \
    conllu_quick_fix

from edparser.components.parsers.biaffine_parser import BiaffineTransformerDependencyParser, \
    BiaffineTransformerSemanticDependencyParser
from edparser.utils.io_util import load_json, get_resource
from edparser.utils.log_util import init_logger
from iwpt2020 import cdroot


def load_conll(path):
    with open(path) as src:
        text = src.read()
        sents = text.split('\n\n')
        sents = [x for x in sents if x.strip()]
        return sents


def combine(folders, file, out):
    sents = []
    for f in folders:
        f = glob.glob(f'{f}/*{file}')
        if not f:
            continue
        assert len(f) == 1
        f = f[0]
        sents += load_conll(f)
    with open(out, 'w') as out:
        out.write('\n\n'.join(sents))
        out.write('\n\n')


def run(lang, do_train=True, do_eval=True, mbert=True):
    """
    Run training and decoding
    :param lang: Language code, 2 letters.
    :param do_train: Train model or not.
    :param do_eval: Evaluate performance (generating output) or not.
    :param mbert: Use mbert or language specific transformers.
    """
    dataset = f'data/iwpt2020/train-dev-combined/{lang}'
    trnfile = f'{dataset}/train.short.conllu'
    # for idx, sent in enumerate(read_conll(trnfile)):
    #     print(f'\r{idx}', end='')
    devfile = f'{dataset}/dev.short.conllu'
    testfile = f'data/iwpt2020/test-udpipe/{lang}.fixed.short.conllu'
    prefix = 'mbert'
    transformer = 'bert-base-multilingual-cased'
    if not mbert:
        prefix = 'bert'
        if lang == 'sv':
            transformer = "KB/bert-base-swedish-cased"
        if lang == 'ar':
            transformer = "asafaya/bert-base-arabic"
        elif lang == 'en':
            transformer = 'albert-xxlarge-v2'
        elif lang == 'ru':
            transformer = "DeepPavlov/rubert-base-cased"
        elif lang == 'fi':
            transformer = "TurkuNLP/bert-base-finnish-cased-v1"
        elif lang == 'it':
            transformer = "dbmdz/bert-base-italian-cased"
        elif lang == 'nl':
            transformer = "wietsedv/bert-base-dutch-cased"
        elif lang == 'et':
            transformer = get_resource(
                'http://dl.turkunlp.org/estonian-bert/etwiki-bert/pytorch/etwiki-bert-base-cased.tar.gz')
        elif lang == 'fr':
            transformer = 'camembert-base'
        elif lang == 'pl':
            transformer = "dkleczek/bert-base-polish-uncased-v1"
        elif lang == 'sk' or lang == 'bg' or lang == 'cs':
            transformer = get_resource(
                'http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt.tar.gz')
        else:
            prefix = 'mbert'
    save_dir = f'data/model/iwpt2020/{lang}/{prefix}_dep'
    # if do_train and os.path.isdir(save_dir):
    #     return
    parser = BiaffineTransformerDependencyParser()
    if do_train and not os.path.isdir(save_dir):
        parser.fit(trnfile,
                   devfile,
                   save_dir,
                   transformer,
                   batch_size=4096,
                   warmup_steps_ratio=.1,
                   samples_per_batch=150,
                   # max_samples_per_batch=75,
                   transformer_dropout=.33,
                   learning_rate=2e-3,
                   learning_rate_transformer=1e-5,
                   # max_seq_length=512,
                   # epochs=1
                   )
    logger = init_logger(name='test', root_dir=save_dir, mode='w')
    parser.config.tree = 'mst'
    # dep_dev_output = f'{save_dir}/{os.path.basename(devfile.replace(".conllu", ".dep.pred.conllu"))}'
    # if not os.path.isfile(dep_dev_output) or do_eval:
    #     parser.evaluate(devfile, save_dir, warm_up=False, output=dep_dev_output, logger=logger)
    dep_test_output = f'{save_dir}/{os.path.basename(testfile.replace(".conllu", ".dep.pred.conllu"))}'
    if not os.path.isfile(dep_test_output) or do_eval:
        parser.load(save_dir, tree='mst')
        parser.evaluate(testfile, save_dir, warm_up=False, output=dep_test_output, logger=None)
    # score = evaluate(devfile, dep_dev_output)
    # dep_dev_elas = score["ELAS"].f1
    # dep_dev_clas = score["CLAS"].f1
    # logger.info(f'DEP score for {lang}:')
    # logger.info(f'ELAS: {dep_dev_elas * 100:.2f} - CLAS:{dep_dev_clas * 100:.2f}')
    if do_train:
        print(f'Model saved in {save_dir}')

    save_dir = f'data/model/iwpt2020/{lang}/{prefix}_sdp'
    parser = BiaffineTransformerSemanticDependencyParser()
    if do_train and not os.path.isdir(save_dir):
        parser.fit(trnfile,
                   devfile,
                   save_dir,
                   transformer,
                   batch_size=1000 if lang == 'cs' else 3000,
                   warmup_steps_ratio=.1,
                   samples_per_batch=150,
                   # max_samples_per_batch=150,
                   transformer_dropout=.33,
                   learning_rate=2e-3,
                   learning_rate_transformer=1e-5,
                   # max_seq_length=512,
                   # epochs=1
                   )
    # (sdp_dev_elas, final_sdp_dev_output), (ensemble_dev_elas, final_ensemble_dev_output) = \
    #     eval_sdp_and_ensemble(parser, devfile, dep_dev_output, save_dir, lang, logger)
    (sdp_test_elas, final_sdp_test_output), (ensemble_test_elas, final_ensemble_test_output) = \
        eval_sdp_and_ensemble(parser, testfile, dep_test_output, save_dir, lang, logger, do_eval)
    save_dir = f'data/model/iwpt2020/{lang}/'
    # copyfile(dep_dev_output, save_dir + 'dev.dep.conllu')
    # copyfile(final_sdp_dev_output, save_dir + 'dev.sdp.conllu')
    # copyfile(final_ensemble_dev_output, save_dir + 'dev.ens.conllu')
    # dev_scores = [dep_dev_elas, sdp_dev_elas, ensemble_dev_elas]
    # winner = max(dev_scores)
    # widx = dev_scores.index(winner)
    dep_test_output = merge_long_sent(dep_test_output)
    evaluate(f'data/iwpt2020/test-udpipe/{lang}.fixed.conllu', dep_test_output)
    dep_test_output = dep_test_output.replace('.conllu', '.fixed.conllu')
    # if widx == 0:
    #     # dep wins, but we don't have output for dep, so let's do it below
    #     best_test_output = dep_test_output
    #     best_task = 'dep'
    # elif widx == 1:
    #     # sdp wins
    #     best_test_output = final_sdp_test_output
    #     best_task = 'sdp'
    # else:
    #     # ensemble wins
    #     best_test_output = final_ensemble_test_output
    #     best_task = 'ens'
    #
    # info = {
    #     'best_task': best_task,
    #     'dev_scores': dict((x, y) for x, y in zip(['dep', 'sdp', 'ens'], dev_scores))
    # }
    # save_json(info, save_dir + 'scores.json')
    # copyfile(best_test_output, save_dir + lang + '.conllu')
    # dev_json = 'data/model/iwpt2020/dev.json'
    # try:
    #     total = load_json(dev_json)
    # except FileNotFoundError:
    #     total = {}
    # total[lang] = info
    # save_json(total, dev_json)

    final_root = f'data/model/iwpt2020/{prefix}'
    dep_root = f'{final_root}/dep'
    sdp_root = f'{final_root}/sdp'
    ens_root = f'{final_root}/ens'
    outputs = [dep_test_output, final_sdp_test_output, final_ensemble_test_output]
    folders = [dep_root, sdp_root, ens_root]
    for o, f in zip(outputs, folders):
        os.makedirs(f, exist_ok=True)
        tmp = f'/home/hhe43/tmp/{lang}.conllu'
        copyfile(o, tmp)
        remove_complete_edges(tmp, tmp)
        restore_collapse_edges(tmp, tmp)
        conllu_quick_fix(tmp, f'{f}/{lang}.conllu')


def merge_long_sent(file, lang=None):
    if not lang:
        lang = os.path.basename(file).split('.')[0]
    long_sent: dict = load_json(f'data/iwpt2020/test-udpipe/{lang}.fixed.long.json')
    long_sent = dict((int(x), y) for x, y in long_sent.items())
    idx = 0
    fout = file.replace('.short', '')
    with open(fout, 'w') as out:
        for sent in load_conll(file):
            long = long_sent.get(idx, None)
            if long:
                out.write(f'{long}\n\n')
                idx += 1
            out.write(f'{sent}\n\n')
            idx += 1
    return fout


def eval_sdp_and_ensemble(parser, devfile, dep_dev_output, save_dir, lang, logger, do_eval=True):
    long_sent: dict = load_json(devfile.replace('.short.conllu', '.long.json'))
    long_sent = dict((int(x), y) for x, y in long_sent.items())
    sdp_dev_output = f'{save_dir}/{os.path.basename(devfile.replace(".conllu", ".sdp.pred.conllu"))}'
    sdp_dev_output = sdp_dev_output.replace('.short', '')
    if not os.path.isfile(sdp_dev_output) or do_eval:
        if not parser.model:
            parser.load(save_dir)
        scores = parser.evaluate(devfile, save_dir, warm_up=False, ret_scores=True, logger=logger,
                                 batch_size=256 if lang == 'cs' else None)[-1]
        sdp_to_dag(parser, scores, sdp_dev_output, long_sent)
    score = evaluate(devfile.replace('.short', ''), sdp_dev_output)
    final_sdp_dev_output = sdp_dev_output.replace('.conllu', '.fixed.conllu')
    sdp_elas = score["ELAS"].f1
    sdp_clas = score["CLAS"].f1
    logger.info(f'SDP score for {lang}:')
    logger.info(f'ELAS: {sdp_elas * 100:.2f} - CLAS:{sdp_clas * 100:.2f}')
    print(f'Model saved in {save_dir}')
    ensemble_output = f'{save_dir}/{os.path.basename(devfile.replace(".conllu", ".ensemble.pred.conllu"))}'
    if not os.path.isfile(sdp_dev_output) or do_eval:
        sdp_to_dag(parser, scores, ensemble_output, long_sent, dep_dev_output)
    score = evaluate(devfile.replace('.short', ''), ensemble_output)
    final_ensemble_output = ensemble_output.replace('.conllu', '.fixed.conllu')
    logger.info(f'Ensemble score for {lang}:')
    ensemble_elas = score["ELAS"].f1
    logger.info(f'ELAS: {ensemble_elas * 100:.2f} - CLAS:{score["CLAS"].f1 * 100:.2f}')
    return (sdp_elas, final_sdp_dev_output), (ensemble_elas, final_ensemble_output)


def sdp_to_dag(parser, scores, output_path, long_sent, dep_dev_output=None):
    # if os.path.isfile(output_path):
    #     return
    with open(output_path, 'w') as out:
        num = 0
        idx = 0
        root_rel_idx = parser.transform.root_rel_idx
        if dep_dev_output:
            trees = CoNLLSentence.from_file(dep_dev_output)
        for arc_scores, rel_scores, mask in scores:
            arc_scores = np.array(arc_scores)
            rel_scores = np.array(rel_scores)
            mask = np.array(mask)
            for a, r, m in zip(arc_scores, rel_scores, mask):
                if dep_dev_output:
                    tree = [0] + [x.head for x in trees[num]]
                    graph = adjust_root_score_then_add_secondary_arcs(a, r, tree, parser.transform.root_rel_idx)
                    # for i, es in enumerate(graph):
                    #     if not i:
                    #         continue
                    #     graph[i] = [(head, trees[num][i - 1].deprel if head == trees[num][i - 1].head else rel) for
                    #                 (head, rel) in es]
                else:
                    tree, graph = mst_then_greedy(a, r, m, root_rel_idx,
                                                  parser.transform.rel_vocab.token_to_idx.get('rel', None))
                sent = CoNLLSentence()
                for i, (t, g) in enumerate(zip(tree, graph)):
                    if not i:
                        continue
                    rels = [x[1] if isinstance(x[1], str) else parser.transform.rel_vocab.idx_to_token[x[1]] for x in g]
                    heads = [x[0] for x in g]
                    if dep_dev_output:
                        head = trees[num][i - 1].head
                        index = heads.index(head)
                        deprel = trees[num][i - 1].deprel
                    else:
                        head = tree[i]
                        index = heads.index(head)
                        deprel = rels[index]
                    deprel = deprel.split('>')[-1]
                    if len(heads) >= 2:
                        heads.pop(index)
                        rels.pop(index)
                    deps = '|'.join(f'{h}:{r}' for h, r in zip(heads, rels))
                    sent.append(CoNLLUWord(id=i, form=None, head=head, deprel=deprel, deps=deps))
                long = long_sent.get(idx, None)
                if long:
                    out.write(f'{long}\n\n')
                    idx += 1
                out.write(f'{sent}\n\n')
                num += 1
                idx += 1


def main():
    cdroot()
    fs = sorted(glob.glob('data/iwpt2020/test-blind/*.txt'))
    # total = load_json('data/model/iwpt2020/dev.json')
    for idx, txt in enumerate(fs):
        basename = os.path.basename(txt)
        langcode = basename.split('.')[0]
        print(f'{idx + 1:02d}/{len(fs)} {basename}')
        # if idx + 1 < 13:
        #     continue
        # if langcode != 'ar':
        #     continue
        # if langcode in total:
        #     continue
        # run(langcode, do_train=True, mbert=False, do_eval=False)
        run(langcode, do_train=False, mbert=True, do_eval=True)


if __name__ == '__main__':
    main()
