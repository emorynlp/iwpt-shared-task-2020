# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-24 12:36
from edparser.metrics.parsing.iwpt20_eval import remove_collapse_edges, enhanced_collapse_empty_nodes, \
    restore_collapse_edges

restored = '/home/hhe43/tmp/ru.conllu'
restore_collapse_edges(
    restored,
    restored.replace('.conllu', '.restored.conllu'))
# enhanced_collapse_empty_nodes(restored, '/home/hhe43/hanlp/collapse.conllu')
