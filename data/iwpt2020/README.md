# Data preprocessing

Download treebanks to this foder and unzip them.

```
iwpt2020
├── test-blind
└── train-dev
```

Run the following scripts in order:

1. `iwpt2020/preprocess_testset.py`
2. `iwpt2020/combine_treebanks.py`
3. `iwpt2020/shorten.py`

The final status of this directory should be as follows.

```bash
iwpt2020
├── test-blind
│   ├── ar.txt
│   ├── bg.txt
│   ├── cs.txt
│   ├── en.txt
│   ├── et.txt
│   ├── fi.txt
│   ├── fr.txt
│   ├── it.txt
│   ├── lt.txt
│   ├── lv.txt
│   ├── nl.txt
│   ├── pl.txt
│   ├── ru.txt
│   ├── sk.txt
│   ├── sv.txt
│   ├── ta.txt
│   └── uk.txt
├── test-udpipe
│   ├── ar.conllu
│   ├── bg.conllu
│   ├── cs.conllu
│   ├── en.conllu
│   ├── et.conllu
│   ├── fi.conllu
│   ├── fr.conllu
│   ├── it.conllu
│   ├── lt.conllu
│   ├── lv.conllu
│   ├── nl.conllu
│   ├── pl.conllu
│   ├── ru.conllu
│   ├── sk.conllu
│   ├── sv.conllu
│   ├── ta.conllu
│   └── uk.conllu
├── train-dev
│   ├── UD_Arabic-PADT
│   ├── UD_Bulgarian-BTB
│   ├── UD_Czech-CAC
│   ├── UD_Czech-FicTree
│   ├── UD_Czech-PDT
│   ├── UD_Czech-PUD
│   ├── UD_Dutch-Alpino
│   ├── UD_Dutch-LassySmall
│   ├── UD_English-EWT
│   ├── UD_English-PUD
│   ├── UD_Estonian-EWT
│   ├── UD_Finnish-PUD
│   ├── UD_Finnish-TDT
│   ├── UD_French-FQB
│   ├── UD_French-Sequoia
│   ├── UD_Italian-ISDT
│   ├── UD_Latvian-LVTB
│   ├── UD_Lithuanian-ALKSNIS
│   ├── UD_Polish-LFG
│   ├── UD_Polish-PDB
│   ├── UD_Polish-PUD
│   ├── UD_Russian-SynTagRus
│   ├── UD_Slovak-SNK
│   ├── UD_Swedish-PUD
│   ├── UD_Swedish-Talbanken
│   ├── UD_Tamil-TTB
│   └── UD_Ukrainian-IU
└── train-dev-combined

31 directories, 34 files
```