import sys
sys.path.append('../../')

import argparse
import pathlib
import tqdm
import torch

from collections import OrderedDict, Counter
from nltk import word_tokenize
from preprocess_utils import get_grams

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

    w2i = build_w2i(movies)
    bow = build_bow(movies, w2i)

    print('done!')
