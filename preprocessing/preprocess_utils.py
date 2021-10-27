import string

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


if __name__ == '__main__':

    print(preprocess_str('Hello! This is an example!'))
