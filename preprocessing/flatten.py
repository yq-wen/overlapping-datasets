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
            utterances = list(map(lambda x: x.strip(), utterances))

            for i in range(1, len(utterances)):

                response_str = utterances[i]
                local_num_contexts = min(i, num_contexts)
                context_lst = utterances[i-local_num_contexts:i]
                context_str = ' __eou__ '.join(context_lst)

                dialogs.append([context_str, response_str])

    return pd.DataFrame(dialogs, columns=['context', 'response'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--output-path')
    parser.add_argument('--num-contexts', type=int, default=1)

    args = parser.parse_args()

    flatten(args.input_path, num_contexts=args.num_contexts).to_csv(args.output_path, index=False)
