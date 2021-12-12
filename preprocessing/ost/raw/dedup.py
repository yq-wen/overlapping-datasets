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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie-dir', type=str, default='samples')

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

    plt.hist(max_overlap.cpu().numpy(), np.arange(0, 1.05, 0.05))
    plt.savefig('movie_overlap.png')

    drop_indices = []
    keep_indices = []
    drop_movie_ids = []
    keep_movie_ids = []

    num_no_overlap_drops = 0

    no_overlap_drop_movie_ids = []

    movie_list = list(movies.items())
    for idx, (movie_id, movie_lines) in tqdm.tqdm(enumerate(movie_list)):
        if max_overlap[idx] < 0.20:
            num_no_overlap_drops += 1
            no_overlap_drop_movie_ids.append(movie_id)
            continue
        elif max_overlap[idx] > 0.80 and idx not in keep_indices:
            drop_indices.append(idx)
            keep_indices.append(max_overlap_idx[idx])
            drop_movie_ids.append(movie_id)
            keep_movie_ids.append(movie_list[max_overlap_idx[idx]][0])

    print('Dropping {} duplicate movies'.format(len(drop_indices)))
    print('Dropping {} movies for too little overlap'.format(num_no_overlap_drops))

    with open('drop_keep.txt', mode='w') as f:
        f.write('drop,keep,\n')
        for i in range(len(keep_movie_ids)):
            f.write('{},{}\n'.format(drop_movie_ids[i], keep_movie_ids[i]))

    with open('no_overlap_drop.txt', mode='w') as f:
        f.write('\n'.join(no_overlap_drop_movie_ids))

    output_dir = pathlib.PosixPath('dedup_' + args.movie_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir()

    for p in movie_dir.glob('*.txt'):
        movie_id = p.name.replace('.txt', '')
        if movie_id in drop_movie_ids or movie_id in no_overlap_drop_movie_ids:
            continue
        shutil.copy(p, output_dir)

    print('done!')
