import itertools

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import CharTokenizer


def split_and_embed(tok, tokens, dev):
    ch_ids, cp_ids = tok.convert_tokens_to_ids(tokens, add_special_tokens=False)
    ptk = tok.pad_token_id
    first = torch.tensor([s[0] for s in ch_ids]).to(dev)
    last = torch.tensor([s[-1] if len(s) > 1 else ptk for s in ch_ids]).to(dev)
    mid = [s[1:-1] if len(s) > 2 else [ptk] for s in ch_ids]
    mid = pad_sequence([torch.tensor(t) for t in mid], batch_first=True).to(dev)
    return first, mid, last


def reassemble(emb, seq_lengths):
    out = []
    start = end = 0
    for sl in seq_lengths:
        end += sl
        out.append(torch.nn.functional.pad(emb[start:end], (0, 0, 1, 1)))  # BOS+...+EOS

        start = end
    return pad_sequence(out, batch_first=True)


class CharEmbedding(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.tok = CharTokenizer()
        self.emb = torch.nn.Embedding(4000, hidden_size, padding_idx=0)
        self.head = torch.nn.Linear(3*hidden_size, output_size)

    def forward(self, seqs_of_tokens):
        dev = next(self.parameters()).device
        all_tokens = list(itertools.chain(*seqs_of_tokens))
        unique_tokens, fwd_i, inv_i = np.unique(all_tokens, return_index=True, return_inverse=True)

        first, mid, last = split_and_embed(self.tok, unique_tokens, dev)
        rep = [self.emb(first), self.emb(mid).sum(dim=1), self.emb(last)]
        out_ = self.head(torch.cat(rep, dim=-1))

        return reassemble(out_[inv_i], [len(s) for s in seqs_of_tokens])
