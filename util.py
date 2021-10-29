import argparse
import random
import pandas as pd

from collections import OrderedDict, namedtuple
from transformers import AutoTokenizer

def build_dd_test_dict(
        num_contexts=1,
        max_num_dialogues=None,
        seed=0,
        path='data/clean_dailydialog/test/dialogues_test.txt'
    ):
    '''Given a path, build a dictionary tha maps contexts to the corresponding
    response
    '''

    test_dict = OrderedDict()

    with open(path, mode='r') as f:
        raw_dialogues = f.readlines()
    num_dialogues = len(raw_dialogues)

    if max_num_dialogues:
        random.seed(seed)
        random.shuffle(raw_dialogues)
        num_dialogues = min(max_num_dialogues, num_dialogues)

    for i in range(num_dialogues):

        # [:-1] because last string is empty after splitting
        utterances = raw_dialogues[i].strip().split('__eou__')[:-1]

        for j in range(len(utterances) - num_contexts):

            context_lst = utterances[j : j + num_contexts]
            context_str = ' '.join(context_lst)
            response_str = utterances[j + num_contexts]
            test_dict[context_str] = [response_str]

    return test_dict

def build_dd_test_dict_from_csv(
        max_num_dialogues=None,
        seed=0,
        path='data/hareesh/df_daily_test_without_duplicates.csv'
    ):
    test_dict = OrderedDict()
    df = pd.read_csv(path)

    num_dialogues = len(df.index)
    if max_num_dialogues:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        num_dialogues = min(max_num_dialogues, num_dialogues)

    for index, row in df.iterrows():
        if index < num_dialogues:
            test_dict[row['line']] = [row['reply']]
        else:
            break

    return test_dict

def build_dd_tests_from_csv(
    path=None,
    max_num_dialogues=None,
    seed=0,
):
    Test = namedtuple('Test', ['score', 'context', 'responses'])
    tests = []

    df = pd.read_csv(path)

    num_dialogues = len(df.index)
    if max_num_dialogues:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        num_dialogues = min(max_num_dialogues, num_dialogues)

    for index, row in df.iterrows():
        if index < num_dialogues:
            test = Test(row['score'], row['eval_context'], [row['eval_response']])
            tests.append(test)
        else:
            break

    return tests

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    test_dict = build_dd_test_dict_from_csv(max_num_dialogues=1000)

    for context, response in test_dict.items():
        print(context, ' | ', response)
