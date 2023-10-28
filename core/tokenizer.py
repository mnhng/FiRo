from .util import _segment


class AutoIncDict(dict):
    def __missing__(self, key):
        self[key] = len(self)
        return self[key]


class Tokenizer():
    def __init__(self,
                unk_token="[UNK]",
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]"
        ):
        self.tok2idx = AutoIncDict()
        self.idx2tok = dict()
        self.pad_token = pad_token
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.sep_token = sep_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.pad_token_id = self.tok2idx[self.pad_token]
        self.bos_token_id = self.tok2idx[self.bos_token]
        self.eos_token_id = self.tok2idx[self.eos_token]
        self.sep_token_id = self.tok2idx[self.sep_token]
        self.unk_token_id = self.tok2idx[self.unk_token]
        self.mask_token_id = self.tok2idx[self.mask_token]

    def __len__(self):
        return len(self.tok2idx)

    def convert_ids_to_tokens(self, ids):
        if len(self.idx2tok) < len(self.tok2idx):
            self.idx2tok.update({v:k for k, v in self.tok2idx.items()})
        return [self.idx2tok.get(i, self.unk_token) for i in ids]

    def convert_tokens_to_ids(self, tokens, add_special_tokens=False, add_new=False):
        if add_new:
            out = [self.tok2idx[t] for t in tokens]
        else:
            out = [self.tok2idx.get(t, self.unk_token_id) for t in tokens]
        if add_special_tokens:
            return [self.bos_token_id] + out + [self.eos_token_id]
        return out

    def tokenize(self, text):
        return _segment(text)

    def batch_encode(self, sequences, add_special_tokens=False, add_new=True):
        return [self.encode(s, add_special_tokens=add_special_tokens, add_new=add_new) for s in sequences]

    def encode(self, text, add_special_tokens=False, add_new=True):
        assert isinstance(text, str)
        return self.convert_tokens_to_ids(self.tokenize(text), add_special_tokens, add_new)

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def decode(self, token_ids):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))


class CharTokenizer():
    def __init__(self):
        self.chr2idx = AutoIncDict()
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token_id = self.chr2idx[self.pad_token]
        self.bos_token_id = self.chr2idx[self.bos_token]
        self.eos_token_id = self.chr2idx[self.eos_token]

    def __len__(self):
        return len(self.chr2idx)

    def convert_tokens_to_ids(self, tokens, add_special_tokens=False):
        tokens = [t[:30] for t in tokens]  # truncate tokens that are too long
        ch_ids = [[self.chr2idx[c] for c in t.lower()] for t in tokens]
        cap_ids = [[int(c.isupper()) for c in t] for t in tokens]
        if add_special_tokens:
            ch_ids = [[self.bos_token_id] + t + [self.eos_token_id] for t in ch_ids]
            cap_ids = [[0] + t + [0] for t in cap_ids]
        return ch_ids, cap_ids

    def tokenize(self, text):
        return _segment(text)

    def batch_encode(self, sequences, add_special_tokens=False):
        return [self.encode(s, add_special_tokens=add_special_tokens) for s in sequences]

    def encode(self, text, add_special_tokens=False):
        assert isinstance(text, str)
        return self.convert_tokens_to_ids(self.tokenize(text), add_special_tokens)
