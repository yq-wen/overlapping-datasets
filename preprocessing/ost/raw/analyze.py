import sys
sys.path.append('../../')

import argparse
import pathlib
import tqdm
import torch
import pickle

import matplotlib.pyplot as plt
import preprocess_utils

from collections import OrderedDict, Counter, namedtuple
from nltk import word_tokenize
from preprocess_utils import get_grams

Movie = namedtuple('Movie', ['movie_id', 'movie_lines'])

def build_w2i(movies):

    counter = Counter()
    for movie_id, movie_lines in tqdm.tqdm(movies.items()):
        for line in movie_lines:
            grams = get_grams(line)
            for gram in grams:
                counter[gram] += 1

    w2i = dict()
    i = 0
    for gram, count in counter.most_common(10000):
        w2i[gram] = i
        i += 1
    return w2i

def build_bow(movies, w2i):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_movies = len(movies)
    vocab_size = len(w2i)

    bow = torch.zeros(num_movies, vocab_size, device=device, dtype=torch.float16)

    for idx, (movie_id, movie_lines) in tqdm.tqdm(enumerate(movies.items())):
        for line in movie_lines:
            grams = get_grams(line)
            for gram in grams:
                if gram in w2i:
                    bow[idx][w2i[gram]] = 1

    return bow

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie-dir', type=str, default='output')

    args = parser.parse_args()

    movie_dir = pathlib.PosixPath(args.movie_dir)

    movies = OrderedDict()

    for p in movie_dir.glob('*.txt'):
        with open(p, mode='r') as f:
            movie_lines = f.readlines()
            movie_lines = list(map(lambda x: x.strip().replace('1 ', ''), movie_lines))
            movies[p.name.replace('.txt', '')] = movie_lines

    num_movies = len(movies)

    w2i = build_w2i(movies)
    bow = build_bow(movies, w2i)
    score_matrix = preprocess_utils.compute_score_matrix(bow, bow)
    score_matrix[range(num_movies), range(num_movies)] = 0
    max_overlap, max_overlap_idx = score_matrix.max(dim=1)
    sorted_max_overlap, sorted_max_overlap_indices = max_overlap.sort()

    plt.hist(max_overlap.cpu().numpy())
    plt.savefig('movie_overlap.png')

    drop_indices = []
    keep_indices = []
    drop_movie_ids = []

    for idx, (movie_id, movie_lines) in tqdm.tqdm(enumerate(movies.items())):
        if max_overlap[idx] > 0.75:
            drop_indices.append(idx)
            keep_indices.append(max_overlap_idx[idx])
            drop_movie_ids.append(movie_id)

    print('Dropping {} movies'.format(len(drop_movie_ids)))

    with open('drop.txt', mode='w') as f:
        f.write('\n'.join(drop_movie_ids))

    print('done!')
