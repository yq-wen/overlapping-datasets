import sys
sys.path.append('..')

import random
import torch

import numpy as np

from tqdm import tqdm
from vector_clean import flatten, df2bow, df2w2i
from design_matrix import DesignMatrix
from preprocess_utils import compute_score_matrix, compute_scores_sep, batch_compute_scores_sep

PATH = '../../data/ijcnlp_dailydialog/dialogues_text.txt'

def random_split(df, num_samples, n=1, random_state=0):
    np.random.seed(random_state)
    selected = np.random.choice(range(df.shape[0]), size=num_samples, replace=False)
    return df.loc[selected], df.drop(selected)

def greedy_split_naive(df, num_samples, n=1, random_state=0):
    '''Greedily select the least overlap
    Arguments:
        df: the dataframe containing all data points
        num_samples: the number of least overlap samples to choose
    Returns:
        new_split_df:
        remaining_df:
    '''
    w2i = df2w2i(df, n=n)

    selected = []

    score_matrix = compute_score_matrix(bow, bow)

    for i in tqdm(range(num_samples)):

        max_values, _ = score_matrix.max(axis=1)
        # mask out already selected samples with inf so they will not be
        # selected again
        max_values[selected] = float('inf')

        min_overlap, selected_idx = max_values.min(axis=0)
        print('min_overlap', min_overlap)

        selected_idx = int(selected_idx)
        assert selected_idx not in selected
        selected.append(selected_idx)

        new_bow[i, :] = bow[selected_idx, :]

    return df.loc[selected].reset_index(drop=True), df.drop(selected).reset_index(drop=True)

def sample_and_prune(df, num_samples, n=1, random_state=0):
    '''Samples more than needed and prune the highly duplicated ones
    Arguments:
        df: the dataframe containing all data points
        num_samples: the number of least overlap samples to choose
    Returns:
        new_split_df:
        remaining_df:
    '''

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    w2i = df2w2i(df, n=n)
    bow = df2bow(df, w2i, n=n)

    starting_num_samples = 20000
    starting_indices = np.random.choice(range(df.shape[0]), size=starting_num_samples, replace=False)

    starting_bow = bow[starting_indices]
    score_matrix = compute_score_matrix(bow, starting_bow)
    score_matrix[starting_indices, range(starting_num_samples)] = 0

    overlap_values, max_overlap_indices = score_matrix.max(dim=0)
    sorted_overlap_values, sorted_indices = overlap_values.sort()

    selected = starting_indices[sorted_indices.cpu()][:num_samples]

    return df.loc[selected].reset_index(drop=True), df.drop(selected).reset_index(drop=True)

if __name__ == '__main__':

    df = flatten(PATH)

    test_df, all_minus_test = sample_and_prune(df , 1000, n=1)
    valid_df, train_df = sample_and_prune(all_minus_test , 1000, n=1)

    test_df.to_csv('test.csv', index=False)
    all_minus_test.to_csv('all_minus_test.csv', index=False)

    valid_df.to_csv('valid.csv', index=False)
    train_df.to_csv('train.csv', index=False)

    random_valid_df, random_train_df = random_split(all_minus_test , 1000, n=1)
    random_valid_df.to_csv('random_valid.csv')
    random_train_df.to_csv('random_train.csv')
