# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 15:37
from typing import List

from edparser.common.structure import SerializableDict
from edparser.utils.io_util import get_resource
from edparser.utils.log_util import logger

class CoNLLWord(SerializableDict):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, deprel=None, phead=None,
                 pdeprel=None):
        """CoNLL format template, see http://anthology.aclweb.org/W/W06/W06-2920.pdf

        Parameters
        ----------
        id : int
            Token counter, starting at 1 for each new sentence.
        form : str
            Word form or punctuation symbol.
        lemma : str
            Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        cpos : str
            Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
        pos : str
            Fine-grained part-of-speech tag, where the tagset depends on the treebank.
        feats : str
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or an underscore if not available.
        head : Union[int, List[int]]
            Head of the current token, which is either a value of ID,
            or zero (’0’) if the token links to the virtual root node of the sentence.
        deprel : Union[str, List[str]]
            Dependency relation to the HEAD.
        phead : int
            Projective head of current token, which is either a value of ID or zero (’0’),
            or an underscore if not available.
        pdeprel : str
            Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = id
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = head
        self.deprel = deprel
        self.lemma = lemma
        self.feats = feats
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        if isinstance(self.head, list):
            return '\n'.join('\t'.join(['_' if v is None else v for v in values]) for values in [
                [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                 None if head is None else str(head), deprel, self.phead, self.pdeprel] for head, deprel in
                zip(self.head, self.deprel)
            ])
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  None if self.head is None else str(self.head), self.deprel, self.phead, self.pdeprel]
        return '\t'.join(['_' if v is None else v for v in values])

    @property
    def nonempty_fields(self):
        return list(f for f in
                    [self.form, self.lemma, self.cpos, self.pos, self.feats, self.head, self.deprel, self.phead,
                     self.pdeprel] if f)


class CoNLLUWord(SerializableDict):
    def __init__(self, id, form, lemma=None, upos=None, xpos=None, feats=None, head=None, deprel=None, deps=None,
                 misc=None):
        """CoNLL-U format template, see https://universaldependencies.org/format.html

        Parameters
        ----------
        id : int
            Token counter, starting at 1 for each new sentence.
        form : Union[str, None]
            Word form or punctuation symbol.
        lemma : str
            Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        upos : str
            Universal part-of-speech tag.
        xpos : str
            Language-specific part-of-speech tag; underscore if not available.
        feats : str
            List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
        head : Union[int, List[int]]
            Head of the current token, which is either a value of ID,
            or zero (’0’) if the token links to the virtual root node of the sentence.
        deprel : Union[str, List[str]]
            Dependency relation to the HEAD.
        deps : str
            Projective head of current token, which is either a value of ID or zero (’0’),
            or an underscore if not available.
        misc : str
            Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = id
        self.form = form
        self.upos = upos
        self.xpos = xpos
        self.head = head
        self.deprel = deprel
        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

    def __str__(self):
        head = self.head
        # if isinstance(head, list):
        #     head = head[0]
        deprel = self.deprel
        # if isinstance(deprel, list):
        #     deprel = deprel[0]
        if not isinstance(head, list):
            head = [head]
            deprel = [deprel]
        deps = self.deps
        if not deps:
            deps = '|'.join(f'{h}:{r}' for h, r in zip(head, deprel))
        values = [str(self.id), self.form, self.lemma, self.upos, self.xpos, self.feats,
                  None if self.head is None else str(head[0]), deprel[0], deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

    @property
    def nonempty_fields(self):
        return list(f for f in
                    [self.form, self.lemma, self.upos, self.xpos, self.feats, self.head, self.deprel, self.deps,
                     self.misc] if f)


class CoNLLSentence(list):
    def __init__(self, words=None):
        """A list of ConllWord

        Parameters
        ----------
        words : Sequence[Union[CoNLLWord, CoNLLUWord]]
            words of a sentence
        """
        super().__init__()
        if words:
            self.extend(words)

    def __str__(self):
        return '\n'.join([word.__str__() for word in self])

    @staticmethod
    def from_str(conll: str, conllu=False):
        """
        Build a CoNLLSentence from CoNLL-X format str

        Parameters
        ----------
        conll : str
             CoNLL-X format string
        CoNLL-U : bool
             Convert to CoNLL-U format
        Returns
        -------
        CoNLLSentence

        """
        words: List[CoNLLWord] = []
        prev_id = None
        for line in conll.strip().split('\n'):
            if line.startswith('#'):
                continue
            cells = line.split('\t')
            cells = [None if c == '_' else c for c in cells]
            if '-' in cells[0]:
                continue
            cells[0] = int(cells[0])
            cells[6] = int(cells[6])
            if cells[0] != prev_id:
                words.append(CoNLLUWord(*cells) if conllu else CoNLLWord(*cells))
            else:
                if isinstance(words[-1].head, list):
                    words[-1].head.append(cells[6])
                    words[-1].deprel.append(cells[7])
                else:
                    words[-1].head = [words[-1].head] + [cells[6]]
                    words[-1].deprel = [words[-1].deprel] + [cells[7]]
            prev_id = cells[0]
        return CoNLLSentence(words)

    @staticmethod
    def from_file(path: str, conllu=False):
        """

        Parameters
        ----------
        path
        conllu

        Returns
        -------
        List[CoNLLSentence]

        """
        with open(path) as src:
            return [CoNLLSentence.from_str(x, conllu) for x in src.read().split('\n\n') if x.strip()]


def collapse_enhanced_empty_nodes(sent: list):
    collapsed = []
    for cells in sent:
        if isinstance(cells[0], float):
            id = cells[0]
            head, deprel = cells[8].split(':', 1)
            for x in sent:
                arrows = [s.split(':', 1) for s in x[8].split('|')]
                arrows = [(head, f'{head}:{deprel}>{r}') if h == str(id) else (h, r) for h, r in arrows]
                arrows = sorted(arrows)
                x[8] = '|'.join(f'{h}:{r}' for h, r in arrows)
            sent[head][7] += f'>{cells[7]}'
        else:
            collapsed.append(cells)
    return collapsed


def read_conll(filepath, underline_to_none=False, enhanced_collapse_empty_nodes=False):
    sent = []
    filepath: str = get_resource(filepath)
    if filepath.endswith('.conllu') and enhanced_collapse_empty_nodes is None:
        enhanced_collapse_empty_nodes = True
    with open(filepath, encoding='utf-8') as src:
        for idx, line in enumerate(src):
            if line.startswith('#'):
                continue
            line = line.strip()
            cells = line.split('\t')
            if line and cells:
                if enhanced_collapse_empty_nodes and '.' in cells[0]:
                    cells[0] = float(cells[0])
                    cells[6] = None
                else:
                    if '-' in cells[0] or '.' in cells[0]:
                        # sent[-1][1] += cells[1]
                        continue
                    cells[0] = int(cells[0])
                    try:
                        cells[6] = int(cells[6])
                    except ValueError:
                        cells[6] = 0
                        logger.exception(f'Wrong CoNLL format {filepath}:{idx + 1}\n{line}')
                if underline_to_none:
                    for i, x in enumerate(cells):
                        if x == '_':
                            cells[i] = None
                sent.append(cells)
            else:
                if enhanced_collapse_empty_nodes:
                    sent = collapse_enhanced_empty_nodes(sent)
                yield sent
                sent = []
    if sent:
        if enhanced_collapse_empty_nodes:
            sent = collapse_enhanced_empty_nodes(sent)
        yield sent
