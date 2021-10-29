import sys
sys.path.append('..')

import argparse
import Levenshtein
import string
import pandas as pd
import numpy
import preprocess_utils

import matplotlib.pyplot as plt

from tqdm import tqdm
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import Counter


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

def df2bow(df, w2i):

    num_examples = df.shape[0]
    vocab_size = len(w2i)

    bow = numpy.zeros((num_examples, vocab_size))

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        line = row['context'] + ' ' + row['response']
        line = preprocess_utils.preprocess_str(line)
        tokens = wordpunct_tokenize(line)
        for token in tokens:
            bow[idx, w2i[token]] = 1

    return bow

def df2bows(df, w2i):

    num_examples = df.shape[0]
    vocab_size = len(w2i)

    context_bow = numpy.zeros((num_examples, vocab_size))
    response_bow = numpy.zeros((num_examples, vocab_size))

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        # context
        line = row['context']
        line = preprocess_utils.preprocess_str(line)
        tokens = wordpunct_tokenize(line)
        for token in tokens:
            context_bow[idx, w2i[token]] = 1

        # response
        line = row['response']
        line = preprocess_utils.preprocess_str(line)
        tokens = wordpunct_tokenize(line)
        for token in tokens:
            response_bow[idx, w2i[token]] = 1

    return context_bow, response_bow

if __name__ == '__main__':

    train_df = flatten('../../data/ijcnlp_dailydialog/train/dialogues_train.txt')
    test_df = flatten('../..//data/ijcnlp_dailydialog/test/dialogues_test.txt')
    valid_df = flatten('../../data/ijcnlp_dailydialog/validation/dialogues_validation.txt')

    dfs = [train_df, test_df, valid_df]

    # Build the counter
    counter = Counter()
    for df in dfs:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            line = row['context'] + ' ' + row['response']
            line = preprocess_utils.preprocess_str(line)
            tokens = wordpunct_tokenize(line)
            for token in tokens:
                counter[token] += 1
    vocab_size = len(counter)

    # Build w2i
    w2i = dict()
    idx = 0
    for w in counter:
        w2i[w] = idx
        idx += 1

    train_context_bow, train_response_bow = df2bows(train_df, w2i)
    valid_context_bow, valid_response_bow = df2bows(valid_df, w2i)
    test_context_bow, test_response_bow   = df2bows(test_df, w2i)

    print('vocab_size:', len(w2i))

    valid_scores, valid_max_overlap_indices = preprocess_utils.compute_scores_sep(
        train_context_bow, train_response_bow,
        valid_context_bow, valid_response_bow,
    )

    test_scores, test_max_overlap_indices = preprocess_utils.compute_scores_sep(
        train_context_bow, train_response_bow,
        test_context_bow, test_response_bow,
    )

    preprocess_utils.dump_results(train_df, valid_df, valid_scores, valid_max_overlap_indices, 'valid')
    preprocess_utils.dump_results(train_df, test_df, test_scores, test_max_overlap_indices, 'test')
