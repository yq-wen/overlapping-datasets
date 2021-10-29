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
        if token in string.punctuation:
            continue
        processed_tokens.append(token)
    return ' '.join(processed_tokens)

def compute_score_matrix(bow_1, bow_2):
    '''
    Arguments:
        bow_1: (num_bow_1_samples, vocab_size)
        bow_2: (num_bow_2_samples, vocab_size)
    Returns:
        score_matrix (num_bow_1_samples, num_bow_2_samples): the score matrix
            where the i-jth entry contains the overlap score of the ith sample
            from bow_1 and the jth sample from bow_2
    '''
    num_1_bow_samples = bow_1.shape[0]
    num_2_bow_samples = bow_2.shape[0]

    bow_1_len = bow_1.sum(axis=1)  # (num_bow_1_samples,)
    bow_2_len = bow_2.sum(axis=1)  # (num_bow_2_samples,)

    # len_matrics all have shape: (num_bow_1_samples, num_bow_2_samples)
    bow_1_len_matrix = numpy.broadcast_to(bow_1_len, (num_2_bow_samples, num_1_bow_samples)).T
    bow_2_len_matrix = numpy.broadcast_to(bow_2_len, (num_1_bow_samples, num_2_bow_samples))
    total_len_matrix = bow_1_len_matrix + bow_2_len_matrix

    overlap_matrix = bow_1 @ bow_2.T  # (num_bow_1_samples, num_bow_2_samples)
    score_matrix = 2 * overlap_matrix / total_len_matrix

    # nans happen where both sentences contain only punctuations
    # therefore, consider them to be an exact overlap
    numpy.nan_to_num(score_matrix, copy=False, nan=1.0)

    return score_matrix

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

def compute_bz_scores(train_bow, eval_bow):

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
    epoch=math.ceil(train_bow.shape[0]/bz)
    train_len = train_bow.sum(axis=1)
    eval_len = eval_bow.sum(axis=1)
    total_overlap_indices=np.zeros((epoch, eval_bow.shape[0]))
    total_max_overlap=np.zeros((epoch, eval_bow.shape[0]))

    for ep in range(epoch):
        print("epoch {}".format(ep))
        train_bow_bz=train_bow[ep*bz:(ep+1)*bz,:]

        overlap_bow = train_bow_bz @ eval_bow.T  # (train_size, eval_size)

        max_overlap_indices = overlap_bow.argmax(axis=0)
        max_overlap = overlap_bow[max_overlap_indices, range(eval_bow.shape[0])]
        total_overlap_indices[ep:ep+1,:]=np.copy(max_overlap_indices+ep*bz)
        total_max_overlap[ep:ep+1,:]=np.copy(max_overlap)

    final_max_overlap=total_max_overlap.max(axis=0)
    index=total_max_overlap.argmax(axis=0)
    final_overlap_indices=total_overlap_indices[index, range(eval_bow.shape[0])].astype(int)

    max_train_len = train_len[final_overlap_indices]
    total_len = max_train_len + eval_len
    scores = 2 * final_max_overlap / (total_len)

    return scores, final_overlap_indices

def compute_scores_sep(train_context_bow, train_response_bow, eval_context_bow, eval_response_bow):
    '''Computes the score for context and response seperately
    Arguments:
        bows (numpy array): bag of words representation (num_samples, vocab_size)
    Returns:
        scores (numpy array): overlap scores for each evaluation sample.
            (shape: (num_eval_samples,))
        max_overlap_indices (numpy array): indices of the training samples that
            generated the maximum overlap. (shape: (num_eval_samples,))
    '''
    context_score_matrix = compute_score_matrix(train_context_bow, eval_context_bow)
    response_score_matrix = compute_score_matrix(train_response_bow, eval_response_bow)

    # score_matrix  = (context_score_matrix + response_score_matrix) / 2
    score_matrix  = numpy.minimum(context_score_matrix, response_score_matrix)

    max_overlap_indices = score_matrix.argmax(axis=0)
    scores = score_matrix[max_overlap_indices, range(score_matrix.shape[1])]

    return scores, max_overlap_indices

def dump_results(train_df, eval_df, scores, max_overlap_indices, output_name):

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

    for threshold in [0.00, 0.25, 0.50, 0.60, 0.75, 1.00]:
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
    dump_results(train_df, eval_df, scores, max_overlap_indices, output_name)

if __name__ == '__main__':

    print(preprocess_str('Hello! This is an example!'))
