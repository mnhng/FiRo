import pickle
import re
import random
import string

import numpy as np
import torch


def get_extended_attention_mask(attention_mask, dtype):
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    return (1.0 - extended_attention_mask) * -10000.0


def get_ext_local_attention_mask(attention_mask, nsize, dtype):
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
    max_seq_len = attention_mask.shape[-1]
    row_idx = torch.arange(max_seq_len).view(1, max_seq_len)
    col_idx = torch.arange(max_seq_len).view(max_seq_len, 1)
    local_mask = (row_idx - col_idx).abs() < (nsize + 1)
    mask = extended_attention_mask * local_mask[None, None, :, :].to(attention_mask.device, attention_mask.dtype)
    mask = mask.to(dtype=dtype)  # fp16 compatibility
    return (1.0 - mask) * -10000.0


def _init_weights(module):
    """Initialize the weights"""
    initializer_range = 0.02
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def pad_mask(sequences, pad=0, batch_first=False):
    ret = [torch.ones(len(s)+pad) for s in sequences]
    return torch.nn.utils.rnn.pad_sequence(ret, batch_first=batch_first)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _segment(text):
    return re.sub(r'([\[\]\'@#.,!?():\\\/$&;"*\-_%])', r' \1 ', text).split()


def pkl_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj


def process_filetype(filetype):
    insert = (filetype // 1000) % 2 == 1
    delete = (filetype // 100) % 2 == 1
    substitute = (filetype // 10) % 2 == 1
    swap = filetype % 2 == 1
    return insert, delete, substitute, swap


def get_all_edit_dist_one(word, filetype=1111, sub_restrict=None):
    """
    Allowable edit_dist_one perturbations:
        1. Insert any lowercase characer at any position other than the start
        2. Delete any character other than the first one
        3. Substitute any lowercase character for any other lowercase letter other than the start
        4. Swap adjacent characters
    We also include the original word. Filetype determines which of the allowable perturbations to use.
    """
    insert, delete, substitute, swap = process_filetype(filetype)
    #last_mod_pos is last thing you could insert before
    last_mod_pos = len(word) - 1
    ed1 = set()
    for pos in range(1, last_mod_pos + 1):  #can add letters at the end
        if delete and pos < last_mod_pos:
            deletion = word[:pos] + word[pos + 1:]
            ed1.add(deletion)
        if swap and pos < last_mod_pos - 1:
            #swapping thing at pos with thing at pos + 1
            swaped = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
            ed1.add(swaped)
        for letter in string.ascii_lowercase:
            if insert:
                #Insert right after pos - 1
                insertion = word[:pos] + letter + word[pos:]
                ed1.add(insertion)
            can_substitute = sub_restrict is None or letter in sub_restrict[
                word[pos]]
            if substitute and pos < last_mod_pos and can_substitute:
                substitution = word[:pos] + letter + word[pos + 1:]
                ed1.add(substitution)
    #Include original word
    ed1.add(word)
    return sorted(list(ed1))
