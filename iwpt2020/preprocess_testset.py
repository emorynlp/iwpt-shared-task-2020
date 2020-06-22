# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-02 15:37
import glob
import json
import os
import requests

from iwpt2020 import cdroot


def parse(text, lang='en'):
    response = requests.post('http://lindat.mff.cuni.cz/services/udpipe/api/process', {
        'data': text,
        'model': lang,
        'tokenizer': True,
        'tagger': True,
        'parser': True,
    })
    assert response.status_code == 200
    data = json.loads(response.text)
    return data['result']


def main():
    print('Preprocess blind test data with UDPipe ...')
    cdroot()
    files = glob.glob('data/iwpt2020/test-blind/*.txt')
    os.makedirs('data/iwpt2020/test-udpipe', exist_ok=True)
    for idx, txt in enumerate(files):
        basename = os.path.basename(txt)
        print(f'{idx + 1}/{len(files)} {basename}')
        lang = basename.split('.')[0]
        with open(txt) as src, open(f'data/iwpt2020/test-udpipe/{lang}.conllu', 'w') as out:
            text = src.read()
            conllu = parse(text, lang)
            out.write(conllu)


if __name__ == '__main__':
    main()
