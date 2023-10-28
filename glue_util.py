import hashlib
import pathlib

import datasets


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def to_words(self):
        exists_b = self.text_b is not None
        split_a = self.text_a.split()
        words = split_a.copy()
        if exists_b:
            split_b = self.text_b.split()
            words.extend(split_b)
        return words, len(split_a)

    def __str__(self):
        if self.text_b:
            return self.text_a + '\t' + self.text_b + '\t' + self.label
        return self.text_a + '\t' + self.label


TASK_TO_KEYS = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}


def get_data(args):
    # Prepare GLUE task
    glue_data = datasets.load_dataset('glue', args.task_name)
    label_list = glue_data['train'].features['label'].names
    train_data = glue_data['train']

    if args.task_name == 'mnli':
        dev_data = glue_data['validation_matched']
    else:
        dev_data = glue_data['validation']

    sentence1_key, sentence2_key = TASK_TO_KEYS[args.task_name]
    train_data = [InputExample(i['idx'], i[sentence1_key], i[sentence2_key] if sentence2_key else None, label_list[i['label']]) for i in train_data]
    dev_data = [InputExample(i['idx'], i[sentence1_key], i[sentence2_key] if sentence2_key else None, label_list[i['label']]) for i in dev_data]
    args.output_mode = 'regression' if args.task_name == 'stsb' or args.task_name == 'sts-b' else 'classification'
    print(f'Train data len: {len(train_data)}, dev data len: {len(dev_data)}')
    return train_data, dev_data, label_list


def cache_filename(args):
    hash_obj = hashlib.new('sha512_256')
    params = [
        args.beam_width,
        args.max_num_words,
        args.model_name_or_path,
        args.task_name
    ]
    for param in params:
        hash_obj.update(str(param).encode())
    return pathlib.Path('cached') / hash_obj.hexdigest()
