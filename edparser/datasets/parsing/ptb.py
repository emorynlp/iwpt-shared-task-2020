# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-02-17 15:46
import os

from edparser.utils.io_util import get_resource, get_exitcode_stdout_stderr
from edparser.utils.log_util import logger

PTB_HOME = 'https://github.com/nikitakit/self-attentive-parser/archive/acl2018.zip#data/'

PTB_TRAIN = PTB_HOME + '02-21.10way.clean'
PTB_VALID = PTB_HOME + '22.auto.clean'
PTB_TEST = PTB_HOME + '23.auto.clean'

PTB_SD330_TRAIN = PTB_HOME + 'train.conllx'
PTB_SD330_VALID = PTB_HOME + 'valid.conllx'
PTB_SD330_TEST = PTB_HOME + 'test.conllx'

PTB_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
}


def convert_to_stanford_dependency_330(src, dst):
    logger.info(f'Converting {os.path.basename(src)} to {os.path.basename(dst)} using Stanford Parser Version 3.3.0. '
                f'It might take a while...')
    sp_home = 'https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip'
    sp_home = get_resource(sp_home)
    # jar_path = get_resource(f'{sp_home}#stanford-parser.jar')
    code, out, err = get_exitcode_stdout_stderr(
        f'java -cp {sp_home}/* edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx '
        f'-treeFile {src}')
    with open(dst, 'w') as f:
        f.write(out)
    if code:
        raise RuntimeError(f'Conversion failed with code {code} for {src}. The err message is:\n {err}'
                           f'Do you have java installed? Do you have enough memory?')


for s, d in zip([PTB_TRAIN, PTB_VALID, PTB_TEST], [PTB_SD330_TRAIN, PTB_SD330_VALID, PTB_SD330_TEST]):
    s = get_resource(s)
    home = os.path.dirname(s)
    d = os.path.join(home, d.split('/')[-1])
    if not os.path.isfile(d):
        convert_to_stanford_dependency_330(s, d)
