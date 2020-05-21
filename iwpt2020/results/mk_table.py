# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-25 01:19

import requests
from bs4 import BeautifulSoup
from sptool.ioutil import load_pickle, save_pickle

from iwpt2020 import cdroot
from iwpt2020.iso639 import ISO639_to_code

online = {
    'mbert_dep': 'https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/eval.pl?team=emorynlp&submid=mbert_dep&dataset=test',
    'mbert_sdp': 'https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/eval.pl?team=emorynlp&submid=mbert_sdp3&dataset=test',
    'mbert_ens': 'https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/eval.pl?team=emorynlp&submid=mbert_ansnop&dataset=test',
    'bert_dep': 'https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/eval.pl?team=emorynlp&submid=bert_dep&dataset=test',
    'bert_sdp': 'https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/eval.pl?team=emorynlp&submid=bert_sdp&dataset=test',
    'bert_ens': 'https://quest.ms.mff.cuni.cz/sharedtask/cgi-bin/eval.pl?team=emorynlp&submid=bert_ens&dataset=test'
}


def safe_float(x):
    if x:
        return float(x)
    return 0


def fetch_all_result(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="lxml")
    lines = soup.find_all('tr')
    ret = {}
    for i, c in enumerate(lines):
        if not i:
            continue
        cells = [x.text for x in c.children]
        Tokens = safe_float(cells[1])
        Sentences = safe_float(cells[3])
        UAS = safe_float(cells[9])
        LAS = safe_float(cells[10])
        CLAS = safe_float(cells[11])
        EULAS = safe_float(cells[14])
        ELAS = safe_float(cells[15])
        if not ELAS:
            ELAS = 0
        ret[cells[0]] = {
            'Tokens': Tokens,
            'Sentences': Sentences,
            'UAS': UAS,
            'LAS': LAS,
            "CLAS": CLAS,
            "EULAS": EULAS,
            'ELAS': ELAS
        }
        if cells[0] == 'Average':
            break
    return ret


def fetch_result(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="lxml")
    lines = soup.find_all('tr')
    ret = {}
    for i, c in enumerate(lines):
        if not i:
            continue
        cells = [x.text for x in c.children]
        score = cells[-1]
        if not score:
            score = 0
        ret[cells[0]] = float(score)
        if cells[0] == 'Average':
            break
    return ret


def main():
    cdroot()
    path = 'all_stat.pkl'
    try:
        stat = load_pickle(path)
    except FileNotFoundError:
        stat = dict((model, fetch_all_result(url)) for model, url in online.items())
        save_pickle(stat, path)
    stat['bert_sdp']['Czech']['UAS'] = 75.95
    stat['bert_sdp']['Czech']['LAS'] = 72.96
    stat['bert_sdp']['Czech']['CLAS'] = 66.06
    stat['bert_sdp']['Czech']['EULAS'] = 69.79
    stat['bert_sdp']['Czech']['ELAS'] = 68.47

    stat['bert_sdp']['Finnish']['UAS'] = 85.89
    stat['bert_sdp']['Finnish']['LAS'] = 84.29
    stat['bert_sdp']['Finnish']['CLAS'] = 80.89
    stat['bert_sdp']['Finnish']['EULAS'] = 80.10
    stat['bert_sdp']['Finnish']['ELAS'] = 79.38

    stat['bert_ens']['Czech']['UAS'] = 86.83
    stat['bert_ens']['Czech']['LAS'] = 83.58
    stat['bert_ens']['Czech']['CLAS'] = 80.84
    stat['bert_ens']['Czech']['EULAS'] = 53.45
    stat['bert_ens']['Czech']['ELAS'] = 51.05

    print(' & ' + ' & '.join('\multicolumn{1}{c|}{\\bf AR}'.replace('AR', ISO639_to_code[lang].upper()) for lang in
                             list(stat['bert_sdp'].keys())[:-1]), end='\\\\\n\hline\hline\n')
    for model in stat.keys():
        if model.endswith('dep'):
            name = 'DTP'
        elif model.endswith('sdp'):
            name = 'DGP'
        else:
            name = 'ENS'
        print(name, end=' & ')
        print(' & '.join(f'{s:.2f}' if s else "-" for s in [x['EULAS'] for x in stat[model].values()][:-1]),
              end=' \\\\\n')
        if name == 'ENS':
            print('\hline')


if __name__ == '__main__':
    main()

# Arabic mbert_ens 67.26
# Bulgarian mbert_ens 88.19
# Czech mbert_ens 85.51
# Dutch mbert_ens 80.72
# English bert_ens 85.3
# Estonian mbert_ens 81.36
# Finnish bert_ens 82.96
# French bert_dep 86.23
# Italian bert_ens 88.52
# Latvian mbert_ens 79.19
# Lithuanian mbert_ens 66.12
# Polish mbert_ens 82.39
# Russian mbert_ens 88.6
# Slovak mbert_ens 82.72
# Swedish mbert_ens 78.19
# Tamil mbert_dep 54.26
# Ukrainian mbert_ens 79.69
