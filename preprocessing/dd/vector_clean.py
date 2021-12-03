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

import preprocess_utils
import matplotlib.pyplot as plt

from tqdm import tqdm
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import Counter
from design_matrix import DesignMatrix
from torch.utils.data import Dataset, DataLoader


def flatten(path, num_contexts=1):
    '''Given the path to the DailyDialog file, convert the dialogs into a
    flattened dataframe
    '''

    # each entry is a list of [context, response]
    dialogs = []

    with open(path, mode='r') as f:

        for dialog in f:

            # [:-1] because last string is empty after splitting
            utterances = dialog.strip().split('__eou__')[:-1]

            for i in range(len(utterances) - num_contexts):

                context_lst = utterances[i : i + num_contexts]
                context_str = ' '.join(context_lst)
                response_str = utterances[i + num_contexts]

                dialogs.append([context_str, response_str])

    return pd.DataFrame(dialogs, columns=['context', 'response'])

def df2w2i(df, n=1):
    # Build the counter
    counter = Counter()
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        line = row['context'] + row['response']
        grams = preprocess_utils.get_grams(line, n=n)
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

def df2bow(df, w2i, n=1):

    num_examples = df.shape[0]
    vocab_size = len(w2i)

    bow = torch.zeros((num_examples, vocab_size), dtype=torch.float16)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        line = row['context']
        grams = preprocess_utils.get_grams(line, n=n)
        for gram in grams:
            bow[idx, w2i[gram]] = 1

    return bow.cuda()

def df2bows(df, w2i, n=1):

    num_examples = df.shape[0]
    vocab_size = len(w2i)

    context_bow = torch.zeros((num_examples, vocab_size), dtype=torch.float16)
    response_bow = torch.zeros((num_examples, vocab_size), dtype=torch.float16)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        # context
        line = row['context']
        grams = preprocess_utils.get_grams(line, n=n)
        for gram in grams:
            context_bow[idx, w2i[gram]] = 1

        # response
        line = row['response']
        line = preprocess_utils.preprocess_str(line)
        grams = preprocess_utils.get_grams(line, n=n)
        for gram in grams:
            response_bow[idx, w2i[gram]] = 1

    return context_bow.cuda(), response_bow.cuda()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Script for cleaning overlaps in DailyDialog')
    parser.add_argument('--n', type=int, default=1, help='order of ngram')
    parser.add_argument('--train-path', type=str, default='../../data/ijcnlp_dailydialog/train/dialogues_train.txt')
    parser.add_argument('--test-path', type=str, default='../../data/ijcnlp_dailydialog/test/dialogues_test.txt')
    parser.add_argument('--valid-path', type=str, default='../../data/ijcnlp_dailydialog/validation/dialogues_validation.txt')

    args = parser.parse_args()

    train_df = flatten(args.train_path)
    train_df.to_csv('train.csv', index=False)
    print('[train], avg # tokens', train_df['context'].apply(lambda x: len(list(preprocess_utils.get_grams(x)))).mean())

    test_df = flatten(args.test_path)
    valid_df = flatten(args.valid_path)

    # train_df = pd.read_csv('../../data/precision_dailydialog/train.csv')
    # test_df = pd.read_csv('../../data/precision_dailydialog/test.csv')
    # valid_df = pd.read_csv('../../data/precision_dailydialog/valid.csv')
    # train_df = pd.read_csv('ost_train.txt', sep='\t')
    # test_df = pd.read_csv('ost_test.txt', sep='\t')
    # valid_df = pd.read_csv('ost_valid.txt', sep='\t')

    dfs = [train_df, test_df, valid_df]

    # Build the counter
    counter = Counter()
    for df in dfs:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            line = row['context'] + ' ' + row['response']
            grams = preprocess_utils.get_grams(line, n=args.n)
            for gram in grams:
                counter[gram] += 1
    vocab_size = len(counter)

    # Build w2i
    w2i = dict()
    idx = 0
    for w in counter:
        w2i[w] = idx
        idx += 1

    print('vocab_size:', len(w2i))

    design_matrix = DesignMatrix(w2i, train_df, n=args.n)

    # test
    test_context_bow, test_response_bow  = df2bows(test_df, w2i, n=args.n)
    test_scores, test_max_overlap_indices = preprocess_utils.batch_compute_scores_sep(
        design_matrix, 8192,
        test_context_bow, test_response_bow,
        verbose=True,
    )
    preprocess_utils.dump_results(train_df, test_df, test_scores, test_max_overlap_indices, 'test')
    del test_context_bow

    # valid
    valid_context_bow, valid_response_bow = df2bows(valid_df, w2i, n=args.n)
    valid_scores, valid_max_overlap_indices = preprocess_utils.batch_compute_scores_sep(
        design_matrix, 8192,
        valid_context_bow, valid_response_bow,
        verbose=True,
    )
    preprocess_utils.dump_results(train_df, valid_df, valid_scores, valid_max_overlap_indices, 'valid')
    del valid_context_bow
