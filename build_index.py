import collections
import string

import numpy as np

import core


def ED1_variants(word, sub_restrict = None):
    #last_mod_pos is last thing you could insert before
    last_mod_pos = len(word)
    ed1 = set()
    for pos in range(1, last_mod_pos + 1): #can add letters at the end
        if pos < last_mod_pos:
            ed1.add(word[:pos] + word[pos + 1:])  # deletion
        if pos < last_mod_pos - 1:
            #swapping thing at pos with thing at pos + 1
            ed1.add(word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:])
        for letter in string.ascii_lowercase:
            ed1.add(word[:pos] + letter + word[pos:])  #Insert right after pos - 1

            can_substitute = sub_restrict is None or letter in sub_restrict[word[pos]]
            if pos < last_mod_pos and can_substitute:
                ed1.add(word[:pos] + letter + word[pos + 1:])  # substitution
    #Include original word
    ed1.add(word)
    return ed1


def inverse_index(word_types):
    ret = collections.defaultdict(set)
    for type_ in word_types:
        for var in ED1_variants(type_):
            ret[var].add(type_)
            # ret[tuple(sorted(var))].add(type_)
    sizes = [len(l) for l in ret.values()]
    print(sum([1 for s in sizes if s > 1]), len(sizes))
    print(f'Neighborhood: {np.mean(sizes)=:.2f} {np.median(sizes)=} {np.max(sizes)=}')
    return {k: sorted(list(v)) for k, v in ret.items()}


def main():
    with open("data/COCA_words_by_freq.txt") as fh:
        types = [l.strip() for l in fh]
    print('all', len(types), 'tokens')
    types = types[:100_000]
    print('used', len(types), 'tokens')

    iidx = inverse_index(types)
    # core.pkl_save(iidx, 'data/index_char.pkl')

    vocab = core.Tokenizer()
    iidx_int = {k: vocab.convert_tokens_to_ids(v, add_new=True) for k, v in iidx.items()}
    core.pkl_save(iidx_int, 'data/index_int.pkl')

    core.pkl_save(vocab, 'data/index_tokenizer.pkl')


if __name__ == '__main__':
    main()
