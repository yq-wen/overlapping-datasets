import argparse
import pathlib
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie-dir', type=str, default='output')
    parser.add_argument('--drop-file', type=str, default='drop.txt', help='file for movies to ignore')
    parser.add_argument('--num-test', type=int, default=700)
    parser.add_argument('--num-valid', type=int, default=700)

    args = parser.parse_args()

    random.seed(0)

    if args.drop_file:
        with open(args.drop_file) as f:
            drop_movie_ids = set(f.read().splitlines())
    else:
        drop_movie_ids = set()

    movie_dir = pathlib.PosixPath(args.movie_dir)
    movie_files = [x for x in movie_dir.iterdir()]
    deduplicated_movie_files = list(filter(lambda x: x.name.replace('.txt', '') not in drop_movie_ids, movie_files))

    assert len(deduplicated_movie_files) + len(drop_movie_ids) == len(movie_files)

    random.shuffle(deduplicated_movie_files)

    splits = {
        'test.txt': deduplicated_movie_files[:args.num_test],
        'valid.txt': deduplicated_movie_files[args.num_test:args.num_test+args.num_valid],
        'train.txt': deduplicated_movie_files[args.num_test+args.num_valid:]
    }

    for split_name, split_movies in splits.items():
        with open(split_name, mode='w') as split_f:
            for movie in split_movies:
                with open(movie, mode='r') as movie_f:
                    split_f.writelines(movie_f.readlines())
