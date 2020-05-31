"""
Command-line arguments for train.py, test.py.

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import argparse


def get_train_args():
    """
    Get arguments needed in train.py.
    """
    parser = argparse.ArgumentParser('Train a model on financial statements data')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs for which to train.')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Learning rate.')

    parser.add_argument('--decay_steps',
                        type=float,
                        default=10**4,
                        help='Learning rate scheduler step size.')

    parser.add_argument('--decay_rate',
                        type=float,
                        default=0.95,
                        help='Decay rate for exponential moving average of parameters.')

    parser.add_argument('--metric_name',
                        type=str,
                        default='mse_loss',
                        choices=('mse_loss',),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--eval_steps',
                        type=int,
                        default=100,
                        help='Number of steps between successive evaluations.')

    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    parser.add_argument('--optimizer',
                        type=str,
                        default='Adadelta',
                        choices=('Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD'),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--use_lr_scheduler',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to use learn rate scheduler.')

    parser.add_argument('--class_weights',
                        type=float,
                        nargs="*",
                        default=[0.59267029, 3.19773635],
                        help='Class weights.')

    args = parser.parse_args()

    return args


def get_test_args():
    """
    Get arguments needed in test.py.
    """
    parser = argparse.ArgumentParser('Test a trained model on financial statements data')

    add_common_args(parser)
    add_train_test_args(parser)

    args = parser.parse_args()

    return args

def get_setup_args():
    """
    Get arguments needed in setup.py.
    """
    parser = argparse.ArgumentParser('Setup financial statements data')

    add_common_args(parser)

    parser.add_argument('--test_size',
                        type=float,
                        default=0.1,
                        help='Test data size as a ratio of the training data.')

    parser.add_argument('--dev_size',
                        type=float,
                        default=0.1,
                        help='Validation data size as a ratio of the remaining training data after splitting for test.')

    args = parser.parse_args()

    return args


def add_common_args(parser):

    """
    Add arguments common to all scripts: train.py, test.py
    """

    parser.add_argument('--input_dir',
                        type=str,
                        default='./data')

    parser.add_argument('--dataset_version',
                        type=str,
                        default='0.0.1')

    parser.add_argument('--company_files',
                        type=str,
                        default='dataset.csv')

    parser.add_argument('--number_of_periods',
                        type=int,
                        default=3)

    parser.add_argument('--download_path',
                        type=str,
                        default='~')

    parser.add_argument('--model',
                        type=str,
                        choices=['baseline', 'lstm'],
                        default='baseline')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--log_device_placement',
                        type=bool,
                        default=False)

    parser.add_argument('--max_sentence_length',
                        type=int,
                        default=512)

    parser.add_argument('--max_document_size',
                        type=int,
                        default=250)

def add_train_test_args(parser):
    """
    Add arguments common to train.py and test.py
    """
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify subdir or test run.')

    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                                  multiple GPUs are available.')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=1024,
                        help='Number of features in the hidden layers.')

    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.4,
                        help='Probability of zeroing an activation in dropout layers.')

    parser.add_argument('--data_shuffle',
                        type=bool,
                        default=False,
                        help='Shuffle the data.')

