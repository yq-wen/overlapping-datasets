import argparse

import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')

    args = parser.parse_args()

    df = pd.read_csv(args.path)
    df['context'] = df['context'].apply(lambda x: x[2:])
    df.to_csv(args.path.replace('.txt', '.csv'))

    print('done...')