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
import random

from tqdm import tqdm
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import Counter
from design_matrix import DesignMatrix
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from vector_clean import flatten, df2w2i, df2bow

def get_dialogs(path):
    dialogs = []
    with open(path, mode='r') as f:
        for line in f:
            dialogs.append(line)
    return dialogs

def get_utterances(path):
    utterances = []
    with open(path, mode='r') as f:
        for line in f:
            ut = line.strip().split('__eou__')[:-1]
            utterances += ut
    return utterances

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

random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Script for deduplicating and splitting dialogues')
    parser.add_argument('--mode', choices=['dedup', 'split', 'hsearch', 'split-sents'], default='split')

    parser.add_argument('--train-path', type=str, default=TRAIN_PATH)
    parser.add_argument('--valid-path', type=str, default=VALID_PATH)
    parser.add_argument('--test-path', type=str, default=TEST_PATH)
    parser.add_argument('--full-path', type=str, default=FULL_PATH)
    parser.add_argument('--segmentation', choices=['dialog', 'utterance'], default='dialog')

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

            score_matrix = preprocess_utils.compute_score_matrix(bow, bow, alpha=1)
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

        sorted_dialogs, sorted_scores = sort_overlap(dialogs, w2i)
        test_dialogs = sorted_dialogs[:NUM_TEST]
        valid_dialogs = sorted_dialogs[NUM_TEST:NUM_TEST+NUM_VALID]
        train_dialogs = sorted_dialogs[NUM_TEST+NUM_VALID:]

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

    elif args.mode == 'split-sents':

        NUM_TEST = 5000
        NUM_VALID = 5000

        df = flatten(args.full_path, num_contexts=1)
        num_samples = df.shape[0]

        w2i = df2w2i(df, n=1)
        bow = df2bow(df, w2i, n=1)

        design_matrix = DesignMatrix(w2i, df, n=1)

        overlap_values, max_overlap_indices = preprocess_utils.batch_compute_self_overlap(design_matrix, 1024, bow, verbose=True)
        # increasing overlap
        sorted_overlap_values, sorted_indices = overlap_values.sort()

        compare_df = pd.DataFrame({
            'score': sorted_overlap_values.cpu().numpy(),
            'train_context': df['context'][max_overlap_indices[sorted_indices].cpu().numpy()].reset_index(drop=True),
            'train_response': df['response'][max_overlap_indices[sorted_indices].cpu().numpy()].reset_index(drop=True),
            'eval_context': df['context'][sorted_indices.cpu().numpy()].reset_index(drop=True),
            'eval_response': df['response'][sorted_indices.cpu().numpy()].reset_index(drop=True),
        })
        compare_df.to_csv('compare.csv')

        # Drop scores with 0 overlap, because they are generally short
        # sentences that only contains very rare words
        modified_df = compare_df[compare_df['score'] != 0].reset_index()

        test_df = pd.DataFrame({
            'context': modified_df['eval_context'][:NUM_TEST],
            'response': modified_df['eval_response'][:NUM_TEST]
        })
        test_df.to_csv('test.csv', index=False)
        print('Test Avg Length:', test_df['context'].apply(lambda x: len(list(preprocess_utils.get_grams(x)))).mean())

        valid_df = pd.DataFrame({
            'context': modified_df['eval_context'][NUM_TEST:NUM_TEST+NUM_VALID],
            'response': modified_df['eval_response'][NUM_TEST:NUM_TEST+NUM_VALID]
        })
        valid_df.to_csv('valid.csv', index=False)
        print('Valid Avg Length:', valid_df['context'].apply(lambda x: len(list(preprocess_utils.get_grams(x)))).mean())

        train_df = pd.DataFrame({
            'context': modified_df['eval_context'][NUM_TEST+NUM_VALID:],
            'response': modified_df['eval_response'][NUM_TEST+NUM_VALID:]
        })
        train_df.to_csv('train.csv', index=False)
        print('Train Avg Length:', train_df['context'].apply(lambda x: len(list(preprocess_utils.get_grams(x)))).mean())

        print('Splitting finished!')

    elif args.mode == 'hsearch':

        alphas = numpy.arange(0, 2.0 + 0.1, 0.1)
        pearsons = []
        spearmans = []
        kendalls = []

        if args.segmentation == 'dialog':

            dialogs = get_dialogs(args.full_path)
            num_dialogs = len(dialogs)

            w2i = build_w2i(dialogs)
            bow = build_bow(dialogs, w2i)

            overlap_matrix, bow_1_len_matrix, bow_2_len_matrix = preprocess_utils.compute_score_matrix_unnormalized(bow, bow)
            len_matrix = bow_1_len_matrix + bow_2_len_matrix

        elif args.segmentation == 'utterance':

            SIZE = 10000

            utterances = get_utterances(args.full_path)
            random.shuffle(utterances)
            utterances = utterances[:SIZE]
            num_dialogs = SIZE

            w2i = build_w2i(utterances)
            bow = build_bow(utterances, w2i)

            overlap_matrix, bow_1_len_matrix, bow_2_len_matrix = preprocess_utils.compute_score_matrix_unnormalized(bow, bow)
            len_matrix = bow_1_len_matrix + bow_2_len_matrix

        for alpha in alphas:

            score_matrix = overlap_matrix / torch.pow(len_matrix, alpha)

            flat_scores = score_matrix.view(-1).cpu()
            flat_len = len_matrix.view(-1).cpu()
            total_samples = num_dialogs * num_dialogs
            chosen_indices = torch.randint(total_samples, (5000,))
            plt.scatter(flat_len[chosen_indices], flat_scores[chosen_indices])

            pearsons.append(pearsonr(flat_len[chosen_indices], flat_scores[chosen_indices]))
            spearmans.append(spearmanr(flat_len[chosen_indices], flat_scores[chosen_indices]))
            kendalls.append(kendalltau(flat_len[chosen_indices], flat_scores[chosen_indices]))

            plt.savefig('alpha={:.2f}.png'.format(alpha))
            plt.close()

            print('Pearson:', pearsons[-1])
            print('Spearman:', spearmans[-1])
            print('Kendall:', kendalls[-1])

        plt.close()
        pearson_coes = list(map(lambda x: x[0], pearsons))
        spearmans_coes = list(map(lambda x: x[0], spearmans))
        kendalls_coes = list(map(lambda x: x[0], kendalls))

        plt.plot(alphas, pearson_coes, label='Pearson')
        plt.plot(alphas, spearmans_coes, label='Spearman')
        plt.plot(alphas, kendalls_coes, label='Kendall')

        plt.legend()
        plt.xlabel('alpha')
        plt.ylabel('coefficient')
        plt.savefig('coefficients_vs_alphas')
