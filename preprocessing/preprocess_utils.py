import string
import numpy

import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize


stopword_set = set(stopwords.words('english'))


def preprocess_str(line):
    '''Proprocessing for the sake of generating a fair bow features.
    '''
    processed_tokens = []
    line = line.lower()
    tokens = wordpunct_tokenize(line)
    for token in tokens:
        if token in stopword_set:
            continue
        elif token in string.punctuation:
            continue
        else:
            processed_tokens.append(token)
    return ' '.join(processed_tokens)


def compute_scores(train_bow, eval_bow):

    '''
    Arguments:
        train_bow (numpy array): bag of words representation for the training
            samples. (shape: (num_train_samples, vocab_size))
        eval_bow (numpy array): bag of words representations for evaluation
            samples to deduplicate. (shape: (num_eval_samples, vocab_size))
    Return:
        scores (numpy array): overlap scores for each evaluation sample.
            (shape: (num_eval_samples,))
        max_overlap_indices (numpy array): indices of the training samples that
            generated the maximum overlap. (shape: (num_eval_samples,))
    '''

    train_len = train_bow.sum(axis=1)
    eval_len = eval_bow.sum(axis=1)
    overlap_bow = train_bow @ eval_bow.T  # (train_size, eval_size)

    max_overlap_indices = overlap_bow.argmax(axis=0)
    max_overlap = overlap_bow[max_overlap_indices, range(eval_bow.shape[0])]
    max_train_len = train_len[max_overlap_indices]
    total_len = max_train_len + eval_len

    scores = 2 * max_overlap / (total_len)

    return scores, max_overlap_indices


def clean_and_dump(train_df, train_bow, eval_df, eval_bow, output_name):
    '''
    Arguments:
        train_df (Series): contains columns for context and response
        eval_df (Series): contains columns for context and response
        train_bow (array): matrix with shape (num_train_examples, vocab_size)
        eval_bow (array): matrix with shape (num_eval_examples, vocab_size)
        output_name (str): root name of all outputs
    '''

    scores, max_overlap_indices = compute_scores(train_bow, eval_bow)

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
    plt.title('Cumulative Word Overlap Distribution')
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

    print(preprocess_str('Hello! This is an example!'))
