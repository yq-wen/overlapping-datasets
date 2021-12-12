import sys
sys.path.append('..')
import argparse
import preprocess_utils

import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--ref-path')
    parser.add_argument('--inp-path')
    parser.add_argument('--out-path')

    args = parser.parse_args()

    ref_df = pd.read_csv(args.ref_path)
    inp_df = pd.read_csv(args.inp_path)

    seen = set()
    drop_indices = []

    for idx, row in inp_df.iterrows():
        text = preprocess_utils.preprocess_str(row['context'] + ' ' + row['response'])
        if text not in seen:
            seen.add(text)
        else:
            drop_indices.append(idx)

    inp_df.drop(drop_indices).to_csv(args.out_path, index=False)

    print('Dropped', len(drop_indices), 'samples')
