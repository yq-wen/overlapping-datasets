import sys
sys.path.append('..')
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import csv
import argparse
import preprocess_utils
from design_matrix import DesignMatrix
import torch

stopword_set = set(stopwords.words('english'))

def df_output(postfix,data_dir):
    output_csv = data_dir + 'df_ost_' + postfix + '.csv'
    with open(data_dir+'src-'+postfix+'.txt') as f1, open(data_dir+'tgt-'+postfix+'.txt') as f2:
        line1=f1.readlines()
        line2=f2.readlines()
        csv_write = csv.writer(open(output_csv, 'w', newline='', encoding='utf-8'))
        csv_head = ["context", "response"]
        csv_write.writerow(csv_head)

        result = []
        count=0
        for idx, src in enumerate(line1):
            tgt=line2[idx]
            d = [src, tgt]
            result.append(d)
            count+=1

    print(count)
    csv_write.writerows(result)



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

def clean_and_dump(train_df, train_bow, eval_df, eval_bow, output_name):
    '''
    Arguments:
        train_df (Series): contains columns for context and response
        eval_df (Series): contains columns for context and response
        train_bow (array): matrix with shape (num_train_examples, vocab_size)
        eval_bow (array): matrix with shape (num_eval_examples, vocab_size)
        output_name (str): root name of all outputs
    '''

    scores, max_overlap_indices = preprocess_utils.compute_bz_scores(train_bow, eval_bow)

    # ----- Plots -----
    bin_width = 0.05
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)

    # Plot a histogram of scores
    plt.hist(scores, bins=bins)
    plt.xlabel('Overlap Ratio', fontsize=20)
    plt.ylabel('# of Samples', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig('{}_scores.png'.format(output_name), bbox_inches='tight')
    plt.close()

    # Plot a cumulative graph of scores
    plt.hist(scores, cumulative=True, density=True, bins=bins)
    plt.xlabel('Overlap Ratio', fontsize=20)
    plt.ylabel('% Data', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig('{}_cumulative_scores.png'.format(output_name), bbox_inches='tight')
    plt.close()

    for threshold in tqdm([0.00, 0.25, 0.50, 0.75, 1.00]):
        drop_indices = (scores > threshold).nonzero()[0]
        print('[{}]: Dropping {} samples for scores>{}'.format(
            output_name,
            len(drop_indices),
            threshold,
        ))
        dropped_df = eval_df.drop(drop_indices)
        dropped_df.to_csv('{}_ost_threshold_{}.csv'.format(output_name, threshold), index=False)

    compare_df = pd.DataFrame({
        'score': scores,
        'train_context': train_df['context'][max_overlap_indices].reset_index(drop=True),
        'train_response': train_df['response'][max_overlap_indices].reset_index(drop=True),
        'eval_context': eval_df['context'].reset_index(drop=True),
        'eval_response': eval_df['response'].reset_index(drop=True),
    })
    compare_df.sort_values('score', inplace=True)
    compare_df.to_csv('{}_ost_compare.csv'.format(output_name))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for cleaning overlaps in DailyDialog')
    parser.add_argument('--n', type=int, default=1, help='order of ngram')

    args = parser.parse_args()
    # df_output('train', '../../data/data_ost/')
    # df_output('valid', '../../data/data_ost/')
    # df_output('test', '../../data/data_ost/')
    train_df = pd.read_csv('../../data/data_ost/df_ost_train.csv')
    valid_df = pd.read_csv('../../data/data_ost/df_ost_valid.csv')
    test_df = pd.read_csv('../../data/data_ost/df_ost_test.csv')

    dfs = [train_df, test_df, valid_df]

    # build the counter
    counter = Counter()
    for df in dfs:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            line = row['context']
            grams = preprocess_utils.get_grams(line, n=args.n)
            for gram in grams:
                counter[gram] += 1
                
    vocab_size = len(counter)

    vocab_size = len(counter)

    # build w2i
    w2i = dict()
    idx = 0
    for w in counter:
        w2i[w] = idx
        idx += 1

    print('vocab_size:', len(w2i))

    design_matrix = DesignMatrix(w2i, train_df, n=args.n)
    test_context_bow = df2bow(test_df, w2i, n=args.n)
    test_scores, test_max_overlap_indices = preprocess_utils.batch_compute_scores_sep(
        design_matrix, 10000,
        test_context_bow, None,
        verbose=True,
    )
    preprocess_utils.dump_results(train_df, test_df, test_scores, test_max_overlap_indices, 'test')
    #clean_and_dump(train_df, train_bow, valid_df, valid_bow, 'valid')
