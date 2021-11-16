import sys
sys.path.append('..')

import argparse
import Levenshtein
import string
import pandas as pd
import numpy
import scipy
import nltk
import torch
import itertools

import preprocess_utils
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import Counter
from design_matrix import DesignMatrix
from torch.utils.data import Dataset, DataLoader

def get_dialogs(path):
    dialogs = []
    with open(path, mode='r') as f:
        for line in f:
            dialogs.append(line)
    return dialogs

def build_w2i(dialogs):
    '''
    Arguments:
        dialogs (list): a list of dialogs, each being a string separated by spaces
    '''
    # Build the counter
    counter = Counter()
    for dialog in dialogs:
        dialog = dialog.replace('__eou__', '')
        grams = preprocess_utils.get_grams(dialog, n=1)
        for gram in grams:
            counter[gram] += 1
    vocab_size = len(counter)

    # Build w2i
    w2i = dict()
    idx = 0
    for w in counter:
        w2i[w] = idx
        idx += 1
    return w2i

def build_bow(dialogs, w2i):
    '''
    Arguments:
        dialogs (list): a list of dialogs, each being a string separated by spaces
    '''
    num_examples = len(dialogs)
    vocab_size = len(w2i)

    bow = torch.zeros((num_examples, vocab_size), dtype=torch.float16).cuda()

    for idx, dialog in enumerate(dialogs):
        dialog = dialog.replace('__eou__', '')
        grams = preprocess_utils.get_grams(dialog, n=1)
        for gram in grams:
            bow[idx, w2i[gram]] = 1
    return bow

TRAIN_PATH = '../../data/ijcnlp_dailydialog/train/dialogues_train.txt'
VALID_PATH = '../../data/ijcnlp_dailydialog/validation/dialogues_validation.txt'
TEST_PATH = '../../data/ijcnlp_dailydialog/test/dialogues_test.txt'
FULL_PATH = '../../data/ijcnlp_dailydialog/dialogues_text.txt'

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Script for deduplicating and splitting dialogues')
    parser.add_argument('--mode', choices=['dedup', 'split'], default='split')

    parser.add_argument('--train-path', type=str, default=TRAIN_PATH)
    parser.add_argument('--valid-path', type=str, default=VALID_PATH)
    parser.add_argument('--test-path', type=str, default=TEST_PATH)
    parser.add_argument('--full-path', type=str, default=FULL_PATH)

    args = parser.parse_args()

    if args.mode == 'dedup':

        train_dialogs = get_dialogs(args.train_path)
        valid_dialogs = get_dialogs(args.valid_path)
        test_dialogs = get_dialogs(args.test_path)

        dialogs = list(itertools.chain(train_dialogs, valid_dialogs, test_dialogs))

        w2i = build_w2i(dialogs)

        train_bow = build_bow(train_dialogs, w2i)
        valid_bow = build_bow(valid_dialogs, w2i)
        test_bow = build_bow(test_dialogs, w2i)

        # Test
        scores, max_overlap_indices = preprocess_utils.compute_scores_sep(
            train_bow, None,
            test_bow, None
        )
        scores = scores.cpu().numpy()
        max_overlap_indices = max_overlap_indices.cpu().numpy()
        preprocess_utils.draw_scores(scores, prefix='test')
        compare_df = pd.DataFrame({
            'score': scores,
            'train_dialogs': [dialogs[i] for i in max_overlap_indices],
            'eval_dialogs' : test_dialogs,
        })
        compare_df.sort_values('score', inplace=True)
        compare_df.to_csv('{}.csv'.format('test'))

        del test_bow

        # Valid
        scores, max_overlap_indices = preprocess_utils.compute_scores_sep(
            train_bow, None,
            valid_bow, None
        )
        scores = scores.cpu().numpy()
        max_overlap_indices = max_overlap_indices.cpu().numpy()
        preprocess_utils.draw_scores(scores, prefix='valid')
        compare_df = pd.DataFrame({
            'score': scores,
            'train_dialogs': [dialogs[i] for i in max_overlap_indices],
            'eval_dialogs' : test_dialogs,
        })
        compare_df.sort_values('score', inplace=True)
        compare_df.to_csv('{}.csv'.format('valid'))

        del valid_bow

        print('done!')

    elif args.mode == 'split':

        NUM_TEST = 1000
        NUM_VALID = 1000

        def sort_overlap(dialogs, w2i):
            '''Retrives N least samples from dialogs
            Return:
                sorted_dialogs
                sorted_scores
            '''
            num_dialogs = len(dialogs)
            bow = build_bow(dialogs, w2i)

            score_matrix = preprocess_utils.compute_score_matrix(bow, bow)
            score_matrix[range(num_dialogs), range(num_dialogs)] = 0

            overlap_values, max_overlap_indices = score_matrix.max(dim=1)
            # increasing overlap
            sorted_overlap_values, sorted_indices = overlap_values.sort()

            sorted_dialogs = []
            sorted_scores = []

            for idx in sorted_indices:
                sorted_dialogs.append(dialogs[idx])
                sorted_scores.append(float(overlap_values[idx]))

            return sorted_dialogs, sorted_scores

        dialogs = get_dialogs(args.full_path)
        w2i = build_w2i(dialogs)

        # Split all dialogs into test and remaining
        sorted_dialogs, sorted_scores = sort_overlap(dialogs, w2i)
        test_dialogs, remaining_dialogs = sorted_dialogs[:NUM_TEST], sorted_dialogs[NUM_TEST:]

        # Split remaining dialogs into valid and train
        sorted_dialogs, sorted_scores = sort_overlap(remaining_dialogs, w2i)
        valid_dialogs, train_dialogs = sorted_dialogs[:NUM_VALID], sorted_dialogs[NUM_VALID:]

        splits = {
            'test': test_dialogs,
            'valid': valid_dialogs,
            'train': train_dialogs
        }

        for split_name, split_dialogs in splits.items():
            with open('{}.txt'.format(split_name), mode='w') as f:
                f.writelines(split_dialogs)
                total_turns = 0
                num_dialogs = 0
                for dialog in split_dialogs:
                    num_dialogs += 1
                    total_turns += dialog.count('__eou__')
                print('{}: {} turns per dialog'.format(split_name, total_turns / num_dialogs))

        print('Splitting finished!')
