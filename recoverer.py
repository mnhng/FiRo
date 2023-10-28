import torch

import core
from glue_util import InputExample


def _split(tokens):
    itokens, group = [], []
    for t in tokens:
        s = core._segment(t)
        itokens.extend(s)
        group.append(len(s))
    return itokens, group


def _merge(tokens, group):
    assert len(tokens) == sum(group)
    start = 0
    ztokens = []
    for l in group:
        ztokens.append(''.join(tokens[start:start+l]))
        start += l
    return ztokens


class Recoverer(object):
    """Clean up a possibly typo-ed string."""
    def __init__(self):
        self.cache = {}

    def recover(self, text):
        """Recover |text| to a new string.

        Used at test time to preprocess possibly typo-ed input.
        """
        if text in self.cache:
            return self.cache[text]
        recovered = self._recover(text)
        self.cache[text] = recovered
        return recovered

    def recover_ex(self, example):
        tokens = example['tokens']
        recovered = self.recover(' '.join(tokens)).split()
        assert len(recovered) == len(tokens), (len(recovered), len(tokens), 'mismatched')
        return {'tokens': recovered, 'ner_tags': example['ner_tags']}

    def _recover(self, text):
        """Actually do the recovery for self.recover()."""
        raise NotImplementedError

    def get_possible_recoveries(self,
                                text,
                                attack_surface,
                                max_num,
                                analyze_res_attacks=False,
                                ret_ball_stats=False):
        """For a clean string, return list of possible recovered strings, or None if too many.

        Used at certification time to exactly compute robust accuracy.

        Returns tuple (list_of_possibilities, num_possibilities)
        where list_of_possibilities is None if num_possibilities > max_num.
        """
        pass

    def recover_example(self, example):
        """Recover an InputExample |example| to a new InputExample.

        Used at test time to preprocess possibly typo-ed input.
        """
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        recovered_tokens = self.recover(' '.join(tokens)).split()
        a_new = ' '.join(recovered_tokens[:a_len])
        if example.text_b:
            b_new = ' '.join(recovered_tokens[a_len:])
        else:
            b_new = None
        return InputExample(example.guid, a_new, b_new, example.label)

    def get_possible_examples(self,
                              example,
                              attack_surface,
                              max_num,
                              analyze_res_attacks=False):
        """For a clean InputExample, return list of InputExample's you could recover to.

        Used at certification time to exactly compute robust accuracy.
        """
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        possibilities, num_poss, perturb_counts = self.get_possible_recoveries(
            ' '.join(tokens),
            attack_surface,
            max_num,
            analyze_res_attacks=analyze_res_attacks)
        if perturb_counts is not None:
            assert len(perturb_counts) == len(possibilities)
        if not possibilities:
            return (None, num_poss)
        out = []
        example_num = 0
        for i in range(len(possibilities)):
            poss = possibilities[i]
            poss_tokens = poss.split()
            a = ' '.join(poss_tokens[:a_len])
            if example.text_b:
                b = ' '.join(poss_tokens[a_len:])
            else:
                b = None
            if not analyze_res_attacks:
                poss_guid = '{}-{}'.format(example.guid, example_num)
            else:
                poss_guid = '{}-{}-{}'.format(example.guid, example_num,
                                              perturb_counts[i])
            out.append(
                InputExample('{}-{}'.format(poss_guid, example_num), a, b,
                             example.label))
            example_num += 1
        return (out, len(out))


class Identity(Recoverer):
    def recover(self, text):
        """Override self.recover() rather than self._recover() to avoid cache."""
        return text


class FiRoRecoverer(Recoverer):
    def __init__(self, idx2int, vocab, firo_network):
        super().__init__()
        self.iidx_int = idx2int
        self.vocab = vocab
        self.net = firo_network

    def _recover(self, text):
        v = self.vocab
        tokens = text.lower().split()
        itokens, group = _split(tokens)
        choices = [self.iidx_int.get(t, [v.unk_token_id]) for t in itokens]

        _, out = self.net([itokens], [choices])
        ids = out[0].cpu().numpy()
        passthrough = True
        if passthrough:
            otokens = v.convert_ids_to_tokens(ids)
            otokens = [y if x == v.unk_token else x for x, y in zip(otokens, itokens)]
        else:
            ids = [v.mask_token_id if id_ == v.unk_token_id else id_ for id_ in ids]
            otokens = v.convert_ids_to_tokens(ids)

        ztokens = _merge(otokens, group)
        assert len(ztokens) == len(tokens)

        return v.convert_tokens_to_string(ztokens)


def initialize(name, rec_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if name == 'identity':
        return Identity()
    elif name == 'firo':
        network = torch.load(rec_path, map_location=device)
        # NOTE: hack to prevent crash because gradient_checkpointing isn't saved with model
        if not hasattr(network.encoder, 'gradient_checkpointing'):
            setattr(network.encoder, 'gradient_checkpointing', False)
        network.eval()
        iidx_int, vocab = core.pkl_load('data/index_int.pkl'), core.pkl_load('data/index_tokenizer.pkl')
        return FiRoRecoverer(iidx_int, vocab, network)
    raise ValueError(name)
