import argparse

import pandas as pd

def flatten(path, num_contexts=1):
    '''Given the path to the DailyDialog file, convert the dialogs into a
    flattened dataframe
    '''

    # each entry is a list of [context, response]
    dialogs = []

    with open(path, mode='r') as f:

        for dialog in f:

            # [:-1] because last string is empty after splitting
            utterances = dialog.strip().split('__eou__')[:-1]

            for i in range(len(utterances) - num_contexts):

                context_lst = utterances[i : i + num_contexts]
                context_str = ' '.join(context_lst)
                response_str = utterances[i + num_contexts]

                dialogs.append([context_str, response_str])

    return pd.DataFrame(dialogs, columns=['context', 'response'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')

    args = parser.parse_args()

    flatten(args.input_path).to_csv(args.output_path, index=False)
