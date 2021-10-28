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


def clean_and_dump(train_df, train_bow, eval_df, eval_bow, output_name):
    '''
    Arguments:
        train_df (Series): contains columns for context and response
        eval_df (Series): contains columns for context and response
        train_bow (array): matrix with shape (num_train_examples, vocab_size)
        eval_bow (array): matrix with shape (num_eval_examples, vocab_size)
        output_name (str): root name of all outputs
    '''

    scores, max_overlap_indices = preprocess_utils.compute_scores(train_bow, eval_bow)

    # ----- Plots -----
    bin_width = 0.05
    bins = numpy.arange(0.0, 1.0 + bin_width, bin_width)

    # Plot a histogram of scores
    plt.hist(scores, bins=bins)
    plt.title('Word Overlap Distribution')
    plt.xlabel('fraction of word overlap')
    plt.ylabel('number of overlapped pairs'.format(output_name))
    plt.savefig('{}_scores.png'.format(output_name))
    plt.close()

    # Plot a cumulative graph of scores
    plt.hist(scores, cumulative=True, density=True, bins=bins)
    plt.title('Cumulative Word Overlap distribution')
    plt.xlabel('fraction of word overlap')
    plt.ylabel('cumulative fraction of the overlapped pairs')
    plt.savefig('{}_cumulative_scores.png'.format(output_name))
    plt.close()

    for threshold in [0.00, 0.25, 0.50, 0.75, 1.00]:
        drop_indices = (scores > threshold).nonzero()[0]
        print('[{}]: Dropping {} samples for scores>{}'.format(
            output_name,
            len(drop_indices),
            threshold,
        ))
        dropped_df = eval_df.drop(drop_indices)
        dropped_df.to_csv('{}_threshold_{}.csv'.format(output_name, threshold), index=False)

    compare_df = pd.DataFrame({
        'score': scores,
        'train_context': train_df['context'][max_overlap_indices].reset_index(drop=True),
        'train_response': train_df['response'][max_overlap_indices].reset_index(drop=True),
        'eval_context': eval_df['context'].reset_index(drop=True),
        'eval_response': eval_df['response'].reset_index(drop=True),
    })
    compare_df.sort_values('score', inplace=True)
    compare_df.to_csv('{}_compare.csv'.format(output_name))

    return


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

    train_bow = df2bow(train_df, w2i)
    valid_bow = df2bow(valid_df, w2i)
    test_bow = df2bow(test_df, w2i)

    print('vocab_size:', len(w2i))

    # ---------- Test Set ----------

    clean_and_dump(train_df, train_bow, test_df, test_bow, 'test')
    clean_and_dump(train_df, train_bow, valid_df, valid_bow, 'valid')

    print(valid_bow)
