# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-20 22:44
import os
from typing import List

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from iwpt2020 import cdroot


def visualize(graph: List[int]):
    G = nx.DiGraph()
    for v, u in enumerate(graph):
        if not v:
            continue
        G.add_edge(u, v)
    return G


def main():
    cdroot()
    graph = [0, 5, 1, 5, 5, 1, 0, 8, 10, 8, 8, 12, 13, 10]
    G = visualize(graph)
    A = to_agraph(G)
    A.layout('dot')
    png = 'data/test/dep.png'
    A.draw(png)
    os.system(f'open {png}')


if __name__ == '__main__':
    main()
