import argparse
import json
import pathlib

import torch

import attacks
import core
import recoverer
import runner
import glue_util


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', required=True)
    parser.add_argument('--output_dir', type=pathlib.Path, required=True)

    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument('--probe_model', required=True, help='model for probing')
    parser.add_argument('--adv_model', required=True, help='adversarially finetuned model')
    parser.add_argument('--max_seq_length', default=128, type=int,
        help=
        'The maximum total input sequence length after tokenization. Sequences longer '
        'than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--do_lower_case', action='store_true', help='use uncased model')

    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--seed', type=int, default=42)

    # Attack parameters
    parser.add_argument('--beam_width', type=int, default=5, help='width for beam search if used...')
    parser.add_argument('--max_num_words', type=int, help='number of words to attack')

    return parser.parse_args()


def main(args):
    args.task_name = args.task_name.lower()

    # Setup
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    core.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get data and model
    _, dev_data, label_list = glue_util.get_data(args)
    recov = recoverer.initialize('identity', None)
    base_rn = runner.initialize(args, recov, label_list, args.probe_model)
    adv_rn = runner.initialize(args, recov, label_list, args.adv_model)

    # clean data evaluation
    results = {'acc': adv_rn.evaluate(dev_data, args.batch_size)}

    # noisy data evaluation
    adv_data = attacks.initialize(args, base_rn).attack_dataset(dev_data)
    results['adv_acc'] = adv_rn.evaluate(adv_data, args.batch_size)

    with open(args.output_dir/f'{args.task_name}_{args.max_num_words}.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(parse_args())
