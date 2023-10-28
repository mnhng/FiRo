import collections
import logging
import os

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_metric

logger = logging.getLogger(__name__)


def compute_metrics(task_name, preds, labels):
    # preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    metric = load_metric("glue", task_name)
    return metric.compute(predictions=preds, references=labels)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, og_text = None, input_text = None, example_idx = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.og_text = og_text
        self.input_text = input_text
        self.example_idx = example_idx


class ModelRunner():
    """Object that can run a model on a given dataset."""
    def __init__(self, recoverer, output_mode, label_list, output_dir, device):
        self.recoverer = recoverer
        self.output_mode = output_mode
        self.label_list = label_list
        self.output_dir = output_dir
        self.device = device

    def train(self, train_data, args):
        """Given already-recovered data, train the model."""
        raise NotImplementedError

    def query(self, examples, batch_size, do_evaluate=True, return_logits=False,
              do_recover=True, use_tqdm=True):
        """Run the recoverer on raw data and query the model on examples."""
        raise NotImplementedError

    def evaluate(self, dataset, batch_size):
        # Need to run recoverer manually on queries
        # Because we're using recoverer.get_possible_examples
        samples = [self.recoverer.recover_example(ex) for ex in dataset]

        preds = self.query(samples, batch_size, do_evaluate=True, do_recover=False)
        id_to_pred = {samples[i].guid: preds[i] for i in range(len(samples))}
        assert len(id_to_pred) == len(samples)

        accuracy = sum(id_to_pred[ex.guid] == ex.label for ex in dataset) / len(dataset)

        return accuracy


class TransformerRunner(ModelRunner):
    def __init__(self, recoverer, output_mode, label_list, output_dir, device, task_name,
                 model_type, model_name_or_path, do_lower_case, max_seq_length):
        super(TransformerRunner, self).__init__(recoverer, output_mode, label_list, output_dir, device)
        self.task_name = task_name
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=len(label_list),
                                            finetuning_task=task_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)
        self.model.to(device)

    def _prep_examples(self, examples, verbose=False):
        features = convert_examples_to_features(
                examples, self.label_list, self.max_seq_length, self.tokenizer, self.output_mode,
                cls_token_at_end=bool(self.model_type in ['xlnet']),  # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                verbose=verbose)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if self.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        all_text_ids = torch.tensor([f.example_idx for f in features], dtype = torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_text_ids)
        return dataset

    def train(self, train_data, args):
        print("Preparing examples.")
        train_dataset = self._prep_examples(train_data)
        print("Starting training.")
        global_step, tr_loss, train_results = train(args, train_dataset, self.model, self.tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        logger.info("Saving model checkpoint to %s", self.output_dir)
        model_to_save = model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        torch.save(args, os.path.join(self.output_dir, 'training_args.bin'))

        # Reload model
        self.load(self.output_dir, self.device)
        print("Finished training.")

    def train2(self, train_data, args):
        train(args, self._prep_examples(train_data), self.model, self.tokenizer)

    def save(self, args):
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        logger.info(f'Saving model checkpoint to {self.output_dir}')
        model_to_save = model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        torch.save(args, self.output_dir/'training_args.bin')

        # Reload model
        self.load(self.output_dir, self.device)

    def load(self, output_dir, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        self.model.to(self.device)

    def query(self, examples, batch_size, do_evaluate=True, return_logits=False,
              do_recover=True, use_tqdm=True):
        if do_recover:
            examples = [self.recoverer.recover_example(x) for x in examples]
        dataset = self._prep_examples(examples)
        eval_sampler = SequentialSampler(dataset)  # Makes sure order is correct
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        example_idxs = None
        self.model.eval()
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = self.model(**inputs)
                inputs['example_idxs'] = batch[4]
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                example_idxs = inputs['example_idxs'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                example_idxs = np.append(example_idxs, inputs['example_idxs'].detach().cpu().numpy(), axis = 0)

        eval_loss = eval_loss / nb_eval_steps
        incorrect_example_indices = None
        if self.output_mode == "classification":
            pred_argmax = np.argmax(preds, axis=1)
            pred_labels = [self.label_list[pred_argmax[i]] for i in range(len(examples))]
            incorrect_example_indices = set(example_idxs[np.not_equal(pred_argmax, out_label_ids)])

        elif self.output_mode == "regression":
            preds = np.squeeze(preds)

        if return_logits:
            return preds
        else:
            return pred_labels


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 verbose = False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    no_oov_examples = 0
    total_examples = 0
    label_distribution = collections.defaultdict(lambda: 0)
    for (ex_index, example) in enumerate(examples):
        total_examples += 1
        if ex_index % 10000 == 0 and verbose:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        exists_oov = False
        a_text = example.text_a
        tokens_a = tokenizer.tokenize(a_text)
        tokens_b = None
        if example.text_b:
            b_text = example.text_b
            tokens_b = tokenizer.tokenize(b_text)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        if not exists_oov:
            no_oov_examples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5 and verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        original_text = example.text_a if not example.text_b else (example.text_a, example.text_b)
        input_text = a_text if not example.text_b else (a_text, b_text)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              og_text = original_text,
                              input_text = input_text,
                              example_idx = ex_index))
        label_distribution[label_id] += 1
    if verbose:
        for label in label_distribution:
            print("Label: {} Percentage: {}".format(label, label_distribution[label] / total_examples))
        #print("Number of examples without oov: {}/{} = {}".format(no_oov_examples, total_examples, no_oov_examples / total_examples))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_results = {}
    for epoch in range(int(args.num_train_epochs)):
        preds = None
        out_label_ids = None
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in pytorch-transformers (see doc)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        results = compute_metrics(args.task_name, preds, out_label_ids)
        train_results[epoch] = results
        print("Train results: ", train_results[epoch])

    #TODO, hacky but saves more significant restructuring...
    args.train_results = train_results
    return global_step, tr_loss / global_step, train_results


def initialize(args, recoverer, label_list, model_name_or_path):
    return TransformerRunner(recoverer, args.output_mode, label_list,
                             args.output_dir, args.device, args.task_name,
                             args.model_type, model_name_or_path,
                             args.do_lower_case, args.max_seq_length)
