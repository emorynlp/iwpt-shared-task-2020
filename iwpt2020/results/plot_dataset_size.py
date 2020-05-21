# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-25 19:15
import math
import numpy as np
from sklearn import linear_model
import requests
from bs4 import BeautifulSoup
from sptool.ioutil import load_pickle, save_pickle
import matplotlib.pyplot as plt
from iwpt2020 import cdroot
from iwpt2020.iso639 import ISO639_to_code
from iwpt2020.results import fetch_result
def load_conll(path):
    with open(path) as src:
        text = src.read()
        sents = text.split('\n\n')
        sents = [x for x in sents if x.strip()]
        return sents

cdroot()
path = 'data/iwpt2020/submission_detail.pkl'
try:
    submission = load_pickle(path)
except FileNotFoundError:
    soup = BeautifulSoup(requests.get('https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/overview.pl').text, 'lxml')
    lines = soup.find_all('tr')
    submission = {}
    for l in lines:
        cells = [x.text for x in l.children]
        if len(cells) != 8:
            continue
        if cells[3] != 'test' or 'baseline' in cells[1] or not cells[7]:
            continue
        submission[cells[1]] = 'https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/' + list(l.children)[7].find('a').get(
            'href')

    submission = dict((k, fetch_result(v)) for k, v in submission.items())
    save_pickle(submission, path)
ranked = sorted([(k, v) for k, v in submission.items()], key=lambda x: x[1]['Average'], reverse=True)


def show(k='English'):
    ranked = sorted([(k, v) for k, v in submission.items()], key=lambda x: x[1][k], reverse=True)
    print(k)
    for team, score in ranked:
        print(f'{team:<20}\t{score[k]:.2f}')
    print()

diff = {}
dataset = {}
for lang in submission['emorynlp'].keys():
    if lang == 'Average':
        continue
    ranked = sorted([(k, v) for k, v in submission.items()], key=lambda x: x[1][lang], reverse=True)
    diff[lang] = submission['emorynlp'][lang] - ranked[0][1][lang]
    code = ISO639_to_code[lang]
    dataset[lang] = math.log10(len(load_conll(f'data/iwpt2020/train-dev-combined/{code}/train.conllu')))

X = np.array(list(dataset.values()))
X = X.reshape(-1, 1)
Y = np.array(list(diff.values()))
Y = Y.reshape(-1, 1)
regr = linear_model.LinearRegression()
regr.fit(X, Y)
pY = regr.predict(X)
plt.scatter(X, Y, color='b')
plt.plot(X, pY, color='r')
plt.xlabel('dataset size')
plt.ylabel('ELAS difference')
plt.savefig('elas_diff_size.pdf')
plt.show()
