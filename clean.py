import argparse
import Levenshtein
import pandas as pd

from tqdm import tqdm

MINDIST = 1

def flatten(path, num_contexts=1):
    '''Given the path to the DailyDialog file, convert the dialogs into a
    flattened csv file
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


def is_dup(store_df, context, response, mindist=MINDIST):

    for idx, row in store_df.iterrows():
        if Levenshtein.distance(row['context'], context) < mindist:
            if Levenshtein.distance(row['response'], response) < mindist:
                return True
    return False

def drop_dupes(input_df, store_df):

    drop_indices = []

    for idx, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
        if is_dup(store_df, row['context'], row['response']):
            drop_indices.append(idx)

    return input_df.drop(drop_indices).reset_index(drop=True)

if __name__ == '__main__':

    # train_f = open('data/ijcnlp_dailydialog/train/dialogues_train.txt', mode='r')
    # test_f = open('data/ijcnlp_dailydialog/test/dialogues_test.txt', mode='r')
    # valid_f = open('data/ijcnlp_dailydialog/validation/dialogues_validation.txt', mode='r')

    # train_lines = set(train_f.readlines())

    # with open('dialogues_test_clean.txt', mode='w') as f:
    #     for line in test_f:
    #         if line in train_lines:
    #             print('skipping:', line)
    #         else:
    #             f.write(line)

    # with open('dialogues_validation_clean.txt', mode='w') as f:
    #     for line in valid_f:
    #         if line in train_lines:
    #             print('skipping:', line)
    #         else:
    #             f.write(line)

    # train_df = flatten('data/ijcnlp_dailydialog/train/dialogues_train.txt')
    # valid_df = flatten('data/ijcnlp_dailydialog/validation/dialogues_validation.txt')
    # test_df = flatten('data/ijcnlp_dailydialog/test/dialogues_test.txt')

    # clean_valid_df = drop_dupes(valid_df, train_df)

    # print(clean_valid_df)

    train_df = pd.read_csv('data/hareesh/df_daily_train.csv')

    context_response_set = set()

    for idx, row in train_df.iterrows():
        context = row['line']
        response = row['reply']
        context_response_str = (context + response).replace(' ', '')
        if context_response_str in context_response_set:
            pass
            # raise ValueError('Train dataset contains duplicates!')
        context_response_set.add(context_response_str)

    test_df = pd.read_csv('data/hareesh/df_daily_test_without_duplicates.csv')


    drop_indices = []
    for idx, row in test_df.iterrows():
        context = row['line']
        response = row['reply']
        context_response_str = (context + response).replace(' ', '')
        if context_response_str in context_response_set:
            print(context, response)
            drop_indices.append(idx)
    print('Number of dups:', len(drop_indices))
    print('Percent duplicates:', len(drop_indices) / test_df.shape[0] * 100)

    clean_test_df = test_df.drop(drop_indices).reset_index(drop=True)
    clean_test_df.to_csv('test.clean.csv', index=False)
