import string
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import csv
import math

stopword_set = set(stopwords.words('english'))
bz=100000

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

def df_output(postfix,data_dir):
    output_csv = data_dir + 'df_ost_' + postfix + '.csv'
    with open(data_dir+'src-'+postfix+'.txt') as f1, open(data_dir+'tgt-'+postfix+'.txt') as f2:
        line1=f1.readlines()
        line2=f2.readlines()
        csv_write = csv.writer(open(output_csv, 'w', newline='', encoding='utf-8'))
        csv_head = ["line", "reply"]
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

def df2bow(df, w2i):

    num_examples = df.shape[0]
    vocab_size = len(w2i)
    bow = np.zeros((num_examples, vocab_size))

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        line = row['line'] + ' ' + row['reply']
        tokens = wordpunct_tokenize(line)

        for token in tokens:
            if token in string.punctuation or token in sw:
                continue
            else:
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

    scores, max_overlap_indices = compute_bz_scores(train_bow, eval_bow)

    # ----- Plots -----
    bin_width = 0.05
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)

    # Plot a histogram of scores
    plt.hist(scores, bins=bins)
    plt.title('Word Overlap Distribution')
    plt.xlabel('fraction of word overlap')
    plt.ylabel('number of overlapped pairs'.format(output_name))
    plt.savefig('{}_ost_scores.png'.format(output_name))
    plt.close()

    # Plot a cumulative graph of scores
    plt.hist(scores, cumulative=True, density=True, bins=bins)
    plt.title('Cumulative Word Overlap Distribution')
    plt.xlabel('fraction of word overlap')
    plt.ylabel('cumulative fraction of the overlapped pairs')
    plt.savefig('{}_ost_cumulative_scores.png'.format(output_name))
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
        'train_context': train_df['line'][max_overlap_indices].reset_index(drop=True),
        'train_response': train_df['reply'][max_overlap_indices].reset_index(drop=True),
        'eval_context': eval_df['line'].reset_index(drop=True),
        'eval_response': eval_df['reply'].reset_index(drop=True),
    })
    compare_df.sort_values('score', inplace=True)
    compare_df.to_csv('{}_ost_compare.csv'.format(output_name))

    return

if __name__ == '__main__':
    df_output('train', '../../data/data_ost/')
    # df_output('valid', '../data/data_ost/')
    # df_output('test', '../data/data_ost/')
    train_df = pd.read_csv('../../data/data_ost/df_ost_train.csv')
    valid_df = pd.read_csv('../../data/data_ost/df_ost_valid.csv')
    test_df = pd.read_csv('../../data/data_ost/df_ost_test.csv')

    dfs = [train_df, test_df, valid_df]
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
    print(train_bow.shape)
    print(valid_bow.shape)
    print(test_bow.shape)
    print(preprocess_str('Hello! This is an example!'))
    #clean_and_dump(train_df, train_bow, test_df, test_bow, 'test')
    clean_and_dump(train_df, train_bow, valid_df, valid_bow, 'valid')
