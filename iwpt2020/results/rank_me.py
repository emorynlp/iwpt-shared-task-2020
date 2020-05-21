# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-25 19:15

import requests
from bs4 import BeautifulSoup
from sptool.ioutil import load_pickle, save_pickle

from iwpt2020 import cdroot
from iwpt2020.results import fetch_result

cdroot()
path = 'data/iwpt2020/submission.pkl'
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
    # place = 0
    # for team, score in ranked:
    #     place += 1
    #     if place > 2:
    #         return
    #     if team == 'emorynlp':
    #         break
    print(k)
    for team, score in ranked:
        print(f'{team:<20}\t{score[k]:.2f}')
    print()


for lang in submission['emorynlp'].keys():
    show(lang)
