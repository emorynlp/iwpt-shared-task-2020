# Enhanced Dependencies Parser

This repository contains code for [*Adaptation of Multilingual Transformer Encoder for Robust Enhanced Universal Dependency Parsing*](https://iwpt20.sigparse.org/pdf/2020.iwpt-1.19.pdf) published to [IWPT 2020](https://iwpt20.sigparse.org/index.html).

 ## Installation

```bash
pip install -e .
```

It requires Python 3.6 or later.

## Data preprocessing

See [data](https://github.com/emorynlp/iwpt-shared-task-2020/tree/master/data/iwpt2020).

## Training and decoding

See https://github.com/emorynlp/iwpt-shared-task-2020/blob/master/iwpt2020/all_in_one.py#L44 .

## Citing

If you use this code for your publication, please cite the original paper:

```latex
@inproceedings{he-choi:2020:iwpt,
	Abstract = {This paper presents our enhanced dependency parsing approach using transformer encoders, coupled with a simple yet powerful ensemble algorithm that takes advantage of both tree and graph dependency parsing. Two types of transformer encoders are compared, a multilingual encoder and language-specific encoders. Our dependency tree parsing (DTP) approach generates only primary dependencies to form trees whereas our dependency graph parsing (DGP) approach handles both primary and secondary dependencies to form graphs. Since DGP does not guarantee the generated graphs are acyclic, the ensemble algorithm is designed to add secondary arcs predicted by DGP to primary arcs predicted by DTP. Our results show that models using the multilingual encoder outperform ones using the language specific encoders for most languages. The ensemble models generally show higher labeled attachment score on enhanced dependencies (ELAS) than the DTP and DGP models. As the result, our best models rank the third place on the macro-average ELAS over 17 languages.},
	Address = {Online},
	Author = {He, Han and Choi, Jinho D.},
	Booktitle = {Proceedings of the 16th International Conference on Parsing Technologies and the IWPT 2020 Shared Task on Parsing into Enhanced Universal Dependencies},
	Month = {July},
	Pages = {181--191},
	Publisher = {Association for Computational Linguistics},
	Title = {Adaptation of Multilingual Transformer Encoder for Robust Enhanced Universal Dependency Parsing},
	Url = {https://www.aclweb.org/anthology/2020.iwpt-1.19},
	Year = {2020},
	Bdsk-Url-1 = {https://www.aclweb.org/anthology/2020.iwpt-1.19}}
```
