import argparse
import itertools
import json
import pathlib
import random
import time

import torch
import transformers as tfm
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset

import core


iidx_int = core.pkl_load('data/index_int.pkl')


def get_neighbors(token, id_=None):
    UNK_ID = 4
    return [UNK_ID] if id_ == UNK_ID else iidx_int.get(token, [UNK_ID])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--out_path', required=True)

    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--local', type=int)
    parser.add_argument('--seed', '-s', type=int, default=999)
    parser.add_argument('--lr', '-l', type=float, default=3e-5)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--layers', type=int, default=1)

    return parser.parse_args()


def _loop(net, batches, opt, sched, args, val_batches, **kwargs):
    best = 0
    for ep in range(1, 1+args.epoch):
        epoch_start = time.time()
        training_loss = _train(net, opt, sched, batches, args.augment)
        print(f'Epoch {ep} | {time.time() - epoch_start:.1f}s : Loss {training_loss:.4f}')

        vacc = _infer(net, val_batches)
        vacc_adv = _infer(net, val_batches, adversarial=True)
        acc = (vacc + vacc_adv) / 2.
        if acc >= best:
            best = acc
            print(f'\t{best=:.4f} {vacc_adv=:.4f}')
            torch.save(net, kwargs['savepath']/'chkpt.pt')
        else:
            print(f'\tfailed {acc=:.4f} {vacc_adv=:.4f}')
        if ep % 5 == 0:
            acc = _infer(net, batches)
            print(f'\ttraining {acc=:.4f}')

    return best


def aug_unified(sentence, no_changes):
    words = sentence.split(" ")
    i_valid = [i for i, w in enumerate(words) if len(w) > 3]
    indices = np.random.choice(i_valid, min(no_changes, len(i_valid)), replace=False)
    for i in indices:
        words[i] = np.random.choice(list(core.get_all_edit_dist_one(words[i])), 1)[0]

    return " ".join(words)


def _train(net, opt, sched, batches, augment):
    losses = []
    net.train()

    for seqs_of_tokens, seqs_of_choices, seqs_of_labels in batches:
        opt.zero_grad()
        loss, _ = net(seqs_of_tokens, seqs_of_choices, seqs_of_labels)
        loss.backward()
        losses.append(loss.item())
        opt.step()

        if augment:
            ids_seqs = [[ch[i] for ch, i in zip(choices, labels)] for choices, labels in zip(seqs_of_choices, seqs_of_labels)]
            # at_seqs_of_tokens = [aug_unified(' '.join(s), no_changes=1000).split() for s in seqs_of_tokens]
            at_seqs_of_tokens = [core._segment(aug_unified(' '.join(s), no_changes=1000)) for s in seqs_of_tokens]
            at_seqs_of_choices = [[get_neighbors(at, id_) for at, id_ in zip(atokens, ids)] for atokens, ids in zip(at_seqs_of_tokens, ids_seqs)]
            at_seqs_of_labels = []
            for achoices, ids in zip(at_seqs_of_choices, ids_seqs):
                at_seqs_of_labels.append([ach.index(id_) for ach, id_ in zip(achoices, ids)])

            opt.zero_grad()
            loss, _ = net(at_seqs_of_tokens, at_seqs_of_choices, at_seqs_of_labels)
            loss.backward()
            opt.step()

        sched.step()

    return sum(losses) / len(losses)


def _infer(net, batches, adversarial=False):
    pred, true, cnt = [], [], []
    for seqs_of_tokens, seqs_of_choices, seqs_of_labels in batches:
        if adversarial:
            # at_seqs_of_tokens = [aug_unified(' '.join(s), no_changes=1000).split() for s in seqs_of_tokens]
            at_seqs_of_tokens = [core._segment(aug_unified(' '.join(s), no_changes=1000)) for s in seqs_of_tokens]
            at_seqs_of_choices = [[get_neighbors(t) for t in tokens] for tokens in at_seqs_of_tokens]

            _, out = net(at_seqs_of_tokens, at_seqs_of_choices)
            pred.extend([s.cpu().numpy() for s in out])
            cnt.extend([[len(choices) for choices in seq] for seq in at_seqs_of_choices])
        else:
            _, out = net(seqs_of_tokens, seqs_of_choices)
            pred.extend([s.cpu().numpy() for s in out])
            cnt.extend([[len(choices) for choices in seq] for seq in seqs_of_choices])

        for s, t in zip(seqs_of_choices, seqs_of_labels):
            true.extend([a1[a2] for a1, a2 in zip(s, t)])
    cnt = np.array(list(itertools.chain(*cnt)))
    true = np.array(true)
    pred = np.concatenate(pred)
    assert len(cnt) == len(true) == len(pred), f'{len(cnt)=}, {len(true)=}, {len(pred)=}'
    plurals = cnt > 1

    return accuracy_score(true[plurals], pred[plurals])


def make_loader(vocab, strings, iidx_int, batch_size, shuffle):
    a, b, c = [], [], []
    for s in strings:
        # tokens = s.split()
        tokens = core._segment(s)
        a.append(tokens)
        ids = vocab.convert_tokens_to_ids(tokens)
        choices = [get_neighbors(token, id_) for token, id_ in zip(tokens, ids)]
        b.append(choices)
        labels = [ch.index(id_) for ch, id_ in zip(choices, ids)]
        c.append(labels)

    data = list(zip(a, b, c))
    return torch.utils.data.DataLoader(data, batch_size, shuffle=shuffle, collate_fn=lambda x: zip(*x))


def load_data(batch_size, vocab):
    TASK_TO_KEYS = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    total_train, total_val, total_test = [], [], []
    for task_name in ['mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']:
        datasets = load_dataset("glue", task_name)
        key1, key2 = TASK_TO_KEYS[task_name]

        train_data = datasets["train"]
        val_data = datasets["test_matched" if task_name == "mnli" else "test"]
        test_data = datasets["validation_matched" if task_name == "mnli" else "validation"]

        if key2:
            train_data = [ex[key1].lower() for ex in train_data] + [ex[key2].lower() for ex in train_data]
            val_data = [ex[key1].lower() for ex in val_data] + [ex[key2].lower() for ex in val_data]
            test_data = [ex[key1].lower() for ex in test_data] + [ex[key2].lower() for ex in test_data]
        else:
            train_data = [ex[key1].lower() for ex in train_data]
            val_data = [ex[key1].lower() for ex in val_data]
            test_data = [ex[key1].lower() for ex in test_data]

        total_train.extend(train_data)
        total_val.extend(val_data)
        total_test.extend(test_data)

        print(f'{task_name} {len(train_data)=:} {len(val_data)=:} {len(test_data)=:}')

    print(f'total {len(train_data)=:} {len(val_data)=:} {len(test_data)=:}')

    batches = make_loader(vocab, total_train, iidx_int, batch_size, shuffle=True)
    vbatches = make_loader(vocab, total_val, iidx_int, batch_size, shuffle=False)
    tbatches = make_loader(vocab, total_test, iidx_int, batch_size, shuffle=False)
    return batches, vbatches, tbatches


def train_mode(args):
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    m_char = core.CharEmbedding(512, 768)
    m_head = core.VarLinear(768)

    if args.local:
        model = core.FiRo(args.layers, m_char, m_head, args.local)
    else:
        model = core.GlobalCorrector(args.layers, m_char, m_head)

    model.to(dev)
    print(f'Local Attention? {args.local}')

    vocab = core.pkl_load('data/index_tokenizer.pkl')
    print(len(vocab))

    batches, vbatches, tbatches = load_data(args.batch_size, vocab)

    opt = tfm.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = tfm.get_constant_schedule_with_warmup(opt, 100)

    out_path = pathlib.Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path/'config.json', 'w') as fh:  # log config
        print(json.dumps(vars(args), sort_keys=True, separators=(',', ': ')), file=fh)

    vacc = _loop(model, batches, opt, sched, args, vbatches, savepath=out_path)
    print(f'{vacc=:.4f}')

    model = torch.load(out_path/'chkpt.pt')

    tacc = _infer(model, tbatches)
    tacc_adv = _infer(model, tbatches, adversarial=True)

    print(f'{tacc=:.4f} {tacc_adv=:.4f}')
    print(out_path)


if __name__ == '__main__':
    args = get_args()

    core.set_seed(args.seed)
    train_mode(args)
