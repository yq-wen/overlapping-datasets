import sys
sys.path.append('../../')

import argparse
import pathlib
import tqdm
import torch
import pickle
import shutil

import numpy as np
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

def dump_movies(movies, p):
    movie_dir = pathlib.PosixPath(args.movie_dir)
    with open(p, mode='w') as f_out:
        for movie_id, movie_text in movies:
            with open(pathlib.PosixPath(movie_dir, movie_id + '.txt'), mode='r') as f_in:
                f_out.write(f_in.read())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie-dir', type=str, default='dedup_samples')
    parser.add_argument('--num-test', type=int, default=220)
    parser.add_argument('--num-valid', type=int, default=90)
    parser.add_argument('--num-train', type=int, default=[1000, 2000, 3000, 4000, 5000], nargs='+')

    args = parser.parse_args()

    movie_dir = pathlib.PosixPath(args.movie_dir)

    movies = OrderedDict()

    for p in movie_dir.glob('*.txt'):
        with open(p, mode='r') as f:
            movie_lines = f.readlines()
            movie_lines = list(map(lambda x: x.strip()[2:].replace('\t', ' '), movie_lines))
            movies[p.name.replace('.txt', '')] = movie_lines

    num_movies = len(movies)

    w2i = build_w2i(movies)
    bow = build_bow(movies, w2i)

    score_matrix = preprocess_utils.compute_score_matrix(bow, bow)
    score_matrix[range(num_movies), range(num_movies)] = 0

    max_overlap, max_overlap_idx = score_matrix.max(dim=1)
    sorted_max_overlap, sorted_max_overlap_indices = max_overlap.sort()

    movie_list = list(movies.items())
    sorted_movies = []

    for idx in sorted_max_overlap_indices:
        sorted_movies.append(movie_list[idx])

    test_movies = sorted_movies[:args.num_test]
    valid_movies = sorted_movies[args.num_test:args.num_test+args.num_valid]
    remaining_movies = sorted_movies[args.num_test+args.num_valid:]

    dump_movies(test_movies, 'test.txt')
    dump_movies(valid_movies, 'valid.txt')

    for num in args.num_train:
        dump_movies(remaining_movies[:num], 'train_{}.txt'.format(num))

    print('splitting done!')
