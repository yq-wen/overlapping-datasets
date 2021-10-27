import argparse
import Levenshtein
import string
import pandas as pd
import numpy

from collections import Counter

from tqdm import tqdm
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


def flatten(path, num_contexts=1):
    '''Given the path to the DailyDialog file, convert the dialogs into a
    flattened csv file
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
        line = row['line'] + ' ' + row['reply']
        tokens = wordpunct_tokenize(line)

        for token in tokens:
            if token in string.punctuation or token in sw:
                continue
            else:
                bow[idx, w2i[token]] = 1

    return bow

def get_score(train_bow, eval_bow):

    train_len = train_bow.sum(axis=1)
    eval_len = eval_bow.sum(axis=1)
    overlap_bow = train_bow @ eval_bow.T  # (train_size, eval_size)

    max_train_indices = overlap_bow.argmax(axis=0)
    max_overlap = overlap_bow[max_train_indices, range(eval_bow.shape[0])]
    max_train_len = train_len[max_train_indices]
    total_len = max_train_len + eval_len

    scores = max_overlap / (total_len)

    return scores

if __name__ == '__main__':

    train_df = pd.read_csv('data/hareesh/df_daily_train.csv')
    test_df = pd.read_csv('data/hareesh/df_daily_test_without_duplicates.csv')
    valid_df = pd.read_csv('data/hareesh/df_daily_valid_without_duplicates.csv')

    dfs = [train_df, test_df, valid_df]
    # dfs = [valid_df]

    sw = stopwords.words('english')

    # build the counter
    counter = Counter()
    for df in dfs:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

            line = row['line'] + ' ' + row['reply']
            tokens = wordpunct_tokenize(line)

            for token in tokens:
                if token in string.punctuation or token in sw:
                    continue
                else:
                    counter[token] += 1

    vocab_size = len(counter)

    # build w2i
    w2i = dict()
    idx = 0
    for w in counter:
        w2i[w] = idx
        idx += 1

    train_bow = df2bow(train_df, w2i)
    valid_bow = df2bow(valid_df, w2i)
    test_bow = df2bow(test_df, w2i)

    test_score = get_score(train_bow, test_bow)

    test_drop_indices = (test_score > 0.30).nonzero()[0]
    print('Dropping {} samples from test'.format(len(test_drop_indices)))

    plt.hist(test_score)
    plt.savefig('test_score.png')
    plt.close()

    test_dedup_df = test_df.drop(test_drop_indices)
    test_dedup_df.to_csv('dedup_0.30_test.csv', index=False)

    print(valid_bow)
