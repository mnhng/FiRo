import copy

import numpy as np
import torch

import core
import glue_util


def pad_mask(sequences, pad=0, batch_first=False):
    ret = [torch.ones(len(s)+pad) for s in sequences]
    return torch.nn.utils.rnn.pad_sequence(ret, batch_first=batch_first)


def save(attacked_data, fpath):
    with open(fpath, 'w') as fh:
        print(*attacked_data, sep='\n', file=fh)


def load(fpath):
    with open(fpath) as fh:
        lines = [l.strip().split('\t') for l in fh]
    attacked_data = []
    for i, row in enumerate(lines):
        if len(row) == 2:
            attacked_data.append(glue_util.InputExample(i, row[0], None, row[1]))
        elif len(row) == 3:
            attacked_data.append(glue_util.InputExample(i, row[0], row[1], row[2]))
    return attacked_data


class Attacker():
    #Going to use attack to cache mapping from clean examples to attack, then run normally in their script.
    def __init__(self,
                 model_runner,
                 eval_batch_size,
                 max_num_words=None,
                 perturbations_per_word=4):

        self.label_map = {
            label: i
            for i, label in enumerate(model_runner.label_list)
        }
        self.model_runner = model_runner
        self.attack = LongDeleteShortAll()
        self.max_num_words = max_num_words
        self.eval_batch_size = eval_batch_size
        self.attacked_count = 0
        self.rng = np.random.RandomState(seed=0)

    def attack_dataset(self, dataset, do_recover=False):
        adv_dataset = []
        for example in dataset:
            adv_dataset.append(self._attack_example(example, max_num_words=self.max_num_words, do_recover=do_recover))
            total_count = len(adv_dataset)
            if total_count % 100 == 0:
                print(f"Performance: successfully attacked {self.attacked_count}/{total_count}")

        return adv_dataset

    def _attack_example(self,
                        clean_example,
                        max_num_words=None,
                        max_attack_attempts=1,
                        do_recover=False
                        ):
        label = clean_example.label
        exists_b = clean_example.text_b is not None
        words, num_in_a = clean_example.to_words()
        if max_num_words is None or max_num_words > len(words):
            max_num_words = len(words)
        perturb_word_idxs = self.rng.choice(len(words),
                                            size=max_num_words,
                                            replace=False)
        to_be_attacked = [words]
        for idx in perturb_word_idxs:
            perturbed_examples = []
            for words in to_be_attacked:
                for prtbd_word in self.attack.get_perturbations(words[idx]):
                    og_copy = words.copy()
                    og_copy[idx] = prtbd_word
                    new_guid = f'{clean_example.guid}-{len(perturbed_examples)}'
                    if not exists_b:
                        perturbed_examples.append(glue_util.InputExample(new_guid, ' '.join(og_copy), label=label))
                    else:
                        perturbed_examples.append(glue_util.InputExample(new_guid, ' '.join(og_copy[:num_in_a]), label=label, text_b=' '.join(og_copy[num_in_a:])))
            #Labels should all be the same, sanity check

            preds = self.model_runner.query(perturbed_examples,
                                            self.eval_batch_size,
                                            do_evaluate=False,
                                            do_recover=do_recover,
                                            return_logits=True,
                                            use_tqdm=False)
            worst_performing_indices, found_incorrect_pred = self._process_preds(
                preds, self.label_map[label])
            if found_incorrect_pred:
                assert len(worst_performing_indices) == 1
                self.attacked_count += 1
                return perturbed_examples[worst_performing_indices[0]]
            else:
                to_be_attacked = [perturbed_examples[i].to_words()[0] for i in worst_performing_indices]
        #Didn't find a successful attack, but still going to do worst case thing...

        return perturbed_examples[worst_performing_indices[0]]

    def _process_preds(self, preds, label):
        #Should return a list of predictions, and whether or not a label is found...
        raise NotImplementedError


class BeamSearchAttacker(Attacker):
    def __init__(self,
                 model_runner,
                 eval_batch_size,
                 beam_width,
                 max_num_words=None):
        super(BeamSearchAttacker, self).__init__(model_runner,
                                                 eval_batch_size,
                                                 max_num_words=max_num_words)
        self.beam_width = beam_width

    def _process_preds(self, preds, label):
        argmax_preds = np.argmax(preds, axis=1)
        if not (argmax_preds == label).all():
            incorrect_idx = np.where(argmax_preds != label)[0][0]
            return [incorrect_idx], True
        if preds.shape[0] <= self.beam_width:
            return list(range(preds.shape[0])), False
        worst_performing_indices = np.argpartition(
            preds[:, label], self.beam_width)[:self.beam_width]
        return list(worst_performing_indices), False


def get_ner_BIO(label_seq):
    tag_list = []
    entity_type = ''
    start = -1
    for i, current_label in enumerate(label_seq + ['O']):  # append ending sentinel token
        current_label = current_label.upper()
        if current_label.startswith('B-'):  # begin of entity
            if entity_type:
                # tag_list.append(f'[{start},{i-1}]{entity_type}')
                tag_list.append((start, i, entity_type))
            entity_type = current_label[2:]
            start = i

        elif current_label.startswith('I-'):  # continuation of entity
            if current_label[2:] == entity_type:  # same tag, go to next position
                continue
            if entity_type: # tags differ, push last entity to list
                # tag_list.append(f'[{start},{i-1}]{entity_type}')
                tag_list.append((start, i, entity_type))
            entity_type = ''
        else:  # O label
            if entity_type:
                # tag_list.append(f'[{start},{i-1}]{entity_type}')
                tag_list.append((start, i, entity_type))
            entity_type = ''

    return tag_list


def extract(tokenizer, token_seq, label_seq, prob_seq=None):
    assert len(token_seq) == len(label_seq)
    assert prob_seq is None or len(token_seq) == len(prob_seq)
    out = []
    entities = get_ner_BIO(label_seq)
    for start, end, entity_type in entities:
        entity_text = tokenizer.convert_tokens_to_string(token_seq[start:end])
        out.append((entity_text, entity_type))

    if prob_seq is not None:
        entity_prob = []
        for start, end, _ in entities:
            entity_prob.append(prob_seq[start:end].log().sum().item())
        return out, entity_prob

    return out, None


class SeqTagAttacker():
    def __init__(self,
                 text_column_name,
                 label_column_name,
                 label_map,
                 model,
                 tokenizer,
                 max_num_words=None,
                 perturbations_per_word=4):
        self.rng = np.random.RandomState(seed=0)
        self.text_col = text_column_name
        self.label_col = label_column_name
        self.label_map = label_map
        self.model = model
        self.tokenizer = tokenizer
        self.max_num_words = max_num_words
        self.attack = LongDeleteShortAll(perturbations_per_word)
        self.attacked_count = 0

    def _prep(self, tokenizer, strings):
        tensors = []
        seqs_of_tokens = []
        for string in strings:
            tokens = tokenizer.tokenize(string, add_special_tokens=True)
            seqs_of_tokens.append(tokens)
            tensors.append(torch.tensor(tokenizer.convert_tokens_to_ids(tokens)))
        return seqs_of_tokens, tensors

    def _predict(self, model, input_):
        dev = next(model.parameters()).device
        with torch.no_grad():
            logits = model(torch.nn.utils.rnn.pad_sequence(input_, batch_first=True).to(dev),
                           attention_mask=pad_mask(input_, batch_first=True).to(dev))[0]
            prob = torch.softmax(logits, dim=2)
            pmax, idx = torch.max(prob, dim=2)
        seqs_of_labels, seqs_of_prob = [], []
        for x, y, z in zip(input_, pmax, idx):
            seqs_of_labels.append([self.label_map[t] for t in z[:len(x)]])
            seqs_of_prob.append(y[:len(x)])
        return seqs_of_labels, seqs_of_prob

    def attack_ex(self, example):
        words = [w.lower() for w in example[self.text_col]]
        voc = self.tokenizer
        truth, _ = extract(voc, words, [self.label_map[t] for t in example[self.label_col]])
        # print('='*80)
        # print(*truth, sep='\n')

        if self.max_num_words is None:
            max_num_words = len(words)
        else:
            max_num_words = min(self.max_num_words, len(words))

        perturb_word_idxs = self.rng.choice(len(words),
                                            size=max_num_words,
                                            replace=False)
        to_be_attacked = [words]
        errors = 0
        for idx in perturb_word_idxs:
            perturbed_examples = []
            for w_ in to_be_attacked:
                for prtbd_word in self.attack.get_perturbations(w_[idx]):
                    perturbed_examples.append(w_[:idx] + [prtbd_word] + w_[idx+1:])

            seqs, input_ = self._prep(voc, [' '.join(s) for s in perturbed_examples])

            seqs_of_labels, seqs_of_prob = self._predict(self.model, input_)

            extr = [extract(voc, *triple) for triple in zip(seqs, seqs_of_labels, seqs_of_prob)]

            worst_idx, errors, likelihood = self._rank(extr, truth)
            # print(errors, '|', ' '.join(perturbed_examples[worst_idx]))
            # print(extr[worst_idx])
            # print('='*80)
            to_be_attacked = [perturbed_examples[worst_idx]]

        assert len(to_be_attacked) == 1
        assert len(to_be_attacked[0]) == len(example[self.text_col])
        adv_example = copy.copy(example)
        adv_example[self.text_col] = to_be_attacked[0]
        self.attacked_count += errors > 0

        return adv_example

    def _rank(self, preds, truth):
        ref = set(truth)
        scores = []
        for pos, (entities, prob) in enumerate(preds):
            e_set = set(entities)
            no_non_overlap = len(ref - e_set) + len(e_set - ref)
            overlap_likelihood = sum([p for e, p in zip(entities, prob) if e in ref])
            scores.append((pos, no_non_overlap, overlap_likelihood))
        # sorted by number of mistakes then by likelihood of overlap
        scores = sorted(sorted(scores, key=lambda x: x[2]), key=lambda x:x[1], reverse=True)
        return scores[0]


class LongDeleteShortAll():
    def __init__(self, perturbations_per_word=4, max_insert_len=4):
        self.cache = {}
        self.perturbations_per_word = perturbations_per_word
        self.max_insert_len = max_insert_len
        self.rng = np.random.RandomState(seed=0)

    def get_perturbations(self, word):
        if word in self.cache:
            return self.cache[word]
        if len(word) > self.max_insert_len:
            perturbations = core.get_all_edit_dist_one(word, filetype=100)  #Just deletions
        else:
            perturbations = core.get_all_edit_dist_one(word)
            if len(perturbations) > self.perturbations_per_word:
                perturbations = self.rng.choice(perturbations, self.perturbations_per_word, replace=False)
        self.cache[word] = perturbations
        return perturbations


def initialize(args, model_runner):
    return BeamSearchAttacker(model_runner, args.batch_size, args.beam_width, max_num_words=args.max_num_words)
