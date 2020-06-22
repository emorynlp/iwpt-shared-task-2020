# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-22 13:22
import glob

from edparser.utils.io_util import get_resource
import os
import shutil

from iwpt2020 import cdroot

cdroot()
iwpt_data = 'data/iwpt2020'
downloaded_iwpt_data = get_resource(
    'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3238/iwpt2020stdata.tgz?sequence=1&isAllowed=y')
if os.path.isdir(iwpt_data):
    shutil.rmtree(iwpt_data, ignore_errors=True)
train_dev = f'{iwpt_data}/train-dev'
os.makedirs(train_dev)
for treebank in glob.glob(f'{downloaded_iwpt_data}/UD_*'):
    shutil.copytree(treebank, f'{train_dev}/{os.path.basename(treebank)}')
shutil.copytree(f'{downloaded_iwpt_data}/test-blind', f'{iwpt_data}/test-blind')

from iwpt2020 import preprocess_testset
preprocess_testset.main()

from iwpt2020 import enhanced_collapse_empty_nodes_for_all
enhanced_collapse_empty_nodes_for_all.main()

from iwpt2020 import combine_treebanks
combine_treebanks.main()

from iwpt2020 import shorten
shorten.main()