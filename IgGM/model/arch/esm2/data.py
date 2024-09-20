# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Sequence, Tuple

import torch

proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X',
             'B', 'U', 'Z', 'O', '.', '-']
}

RawMSA = Sequence[Tuple[str, str]]


class Alphabet(object):

    def __init__(
            self,
            standard_toks: Sequence[str],
            prepend_toks: Sequence[str] = ('<null_0>', '<pad>', '<eos>', '<unk>'),
            append_toks: Sequence[str] = ('<cls>', '<mask>', '<sep>'),
            prepend_bos: bool = True,
            append_eos: bool = False,
            use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)

        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f'<null_{i + 1}>')

        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx['<unk>']
        self.padding_idx = self.get_idx('<pad>')
        self.cls_idx = self.get_idx('<cls>')
        self.mask_idx = self.get_idx('<mask>')
        self.eos_idx = self.get_idx('<eos>')
        self.sep_idx = self.get_idx('<sep>')

        self.nspecial = {self.unk_idx, self.padding_idx, self.cls_idx, self.eos_idx, self.sep_idx}

    def __len__(self):
        return len(self.all_toks)

    def pad(self):
        return self.padding_idx

    def cls(self):
        return self.cls_idx

    def eos(self):
        return self.eos_idx

    def mask(self):
        return self.mask_idx

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_idx_fasta(self, toks):
        return [self.get_idx(s) for s in toks]

    def get_idx_msa(self, toks_msa):
        assert isinstance(toks_msa, list)
        return [self.get_idx_fasta(toks) for toks in toks_msa]

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return {'toks': self.toks}

    def get_batch_converter(self):
        return BatchConverter(self)

    @classmethod
    def from_dict(cls, d, **kwargs):
        return cls(standard_toks=d['toks'], **kwargs)

    @classmethod
    def from_architecture(cls, name: str, ) -> 'Alphabet':
        if name in ('ESM-1', 'protein_bert_base'):
            standard_toks = proteinseq_toks['toks']
            prepend_toks: Tuple[str, ...] = ('<null_0>', '<pad>', '<eos>', '<unk>')
            append_toks: Tuple[str, ...] = ('<cls>', '<mask>', '<sep>')
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ('ESM-1b', 'roberta_large'):
            standard_toks = proteinseq_toks['toks']
            prepend_toks = ('<cls>', '<pad>', '<eos>', '<unk>')
            append_toks = ('<mask>',)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ('MSA Transformer', 'msa_transformer'):
            standard_toks = proteinseq_toks['toks']
            prepend_toks = ('<cls>', '<pad>', '<eos>', '<unk>')
            append_toks = ('<mask>',)
            prepend_bos = True
            append_eos = False
            use_msa = True
        else:
            raise ValueError('Unknown architecture selected')
        return cls(
            standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa
        )


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)

        max_len = max(len(seq_str) for _, seq_str in raw_batch)

        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str) in enumerate(raw_batch):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(
                [self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64
            )
            tokens[
            i,
            int(self.alphabet.prepend_bos): len(seq_str)
                                            + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[
                    i, len(seq_str) + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return labels, strs, tokens
