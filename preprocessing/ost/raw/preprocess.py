import argparse
import os
import pathlib
import multiprocessing
import tqdm

from parlai.tasks.opensubtitles import build_2018
from parlai.utils.io import PathManager

# copied from ParlAI codebase
# modified to only read one subtitle per movie
def get_list_of_files(top_path):
    result = {}
    for path, _dirs, files in os.walk(top_path):
        for filename in files:
            if filename.endswith('.xml'):
                full_filename = os.path.realpath(os.path.join(path, filename))
                assert PathManager.exists(full_filename), 'Bad file ' + full_filename
                movie_id = build_2018.get_movie_id(full_filename)
                if movie_id not in result:
                    result[movie_id] = [full_filename]
    return result

# helper function outputs both the movie id and the processed text
def helper(dict_item):
    movie_id, path = dict_item
    return movie_id, processor(dict_item)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--path', type=str, default='/mnt/4B2738DA6E2B8714/OST/OpenSubtitles')
    parser.add_argument('--path', type=str, default='OpenSubtitles')
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--output-dir', type=str, default='output')

    args = parser.parse_args()

    output_dir = pathlib.PosixPath(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    movies = get_list_of_files(args.path)
    processor = build_2018.DataProcessor(False)

    # use os.cpu_count() for better performance
    with multiprocessing.Pool(processes=2) as pool:
        for movie_id, text in pool.imap(helper, tqdm.tqdm(movies.items())):
            with open(pathlib.PosixPath(output_dir, str(movie_id) + '.txt'), mode='w') as f:
                f.write(text)
