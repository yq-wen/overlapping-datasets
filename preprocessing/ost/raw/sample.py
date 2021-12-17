import argparse
import random
import shutil

from pathlib import PosixPath

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='full-hist')
    parser.add_argument('--num-samples', type=int, default=10000)

    args = parser.parse_args()

    full_dir = PosixPath(args.dir)

    movies = list(full_dir.glob('*.txt'))

    random.seed(1)
    samples = random.sample(movies, args.num_samples)

    output_dir = PosixPath('samples')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    for sample in samples:
        shutil.copy(str(sample), str(output_dir))
