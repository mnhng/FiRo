import argparse
import pathlib

import numpy as np
import torch

import attacks
import core
import recoverer
import runner
import glue_util


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--output_dir', type=pathlib.Path, required=True)

    ## Other parameters
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_lower_case', action='store_true', help='Set this flag if you are using an uncased model.')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument('--num_train_epochs', default=1, type=float)

    parser.add_argument('--overwrite_output_dir', action='store_true', help='Overwrite the content of the output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')

    parser.add_argument( '--recoverer', default='identity', help='Which recovery strategy to use (default: do nothing)')
    parser.add_argument('--max_num_words', type=int, help='Number of words to attack')
    parser.add_argument('--rec_path', help='Path to where recoverer is stored')

    #Attack parameters
    parser.add_argument('--do_attack', action='store_true')
    parser.add_argument('--beam_width', type=int, default=5, help='Width for beam search if used...')
    return parser.parse_args()


def main(args):
    args.task_name = args.task_name.lower()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    core.set_seed(args.seed)
    if args.output_dir.exists() and args.output_dir.iterdir() and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f'{args.output_dir} exists and is not empty. Use --overwrite_output_dir overwrite.')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get data and model
    train_data, dev_data, label_list = glue_util.get_data(args)
    recov = recoverer.initialize(args.recoverer, args.rec_path)
    base_rn = runner.initialize(args, recov, label_list, args.model_name_or_path)

    # Run training and evaluation
    dev_adv_data = attacks.initialize(args, base_rn).attack_dataset(dev_data)
    best_acc = base_rn.evaluate(dev_adv_data, args.batch_size)

    no_chunks = (len(train_data) // 1000) + (len(train_data) % 1000 > 0)
    for i, chunk in enumerate(np.array_split(train_data, no_chunks)):
        with open(args.output_dir/'proc.log', 'a') as fh:
            print(f'adv iter {i} {best_acc:.4f}', file=fh)
        mixed_data = list(chunk) + attacks.initialize(args, base_rn).attack_dataset(chunk)
        base_rn.train2(mixed_data, args)

        acc = base_rn.evaluate(dev_adv_data, args.batch_size)
        if best_acc > acc:
            break
        else:
            best_acc = acc
            base_rn.save(args)


if __name__ == "__main__":
    main(parse_args())
