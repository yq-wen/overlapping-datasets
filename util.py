import argparse

from collections import OrderedDict
from transformers import AutoTokenizer

def build_dd_test_dict(
        tokenizer,
        num_contexts=1,
        path='data/ijcnlp_dailydialog/test/dialogues_test.txt'
    ):
    '''Given a path, build a dictionary tha maps contexts to the corresponding
    response
    '''

    test_dict = OrderedDict()

    with open(path, mode='r') as f:
        raw_dialogues = f.readlines()
    num_dialogues = len(raw_dialogues)

    contexts = []
    responses = []

    for i in range(num_dialogues):

        # [:-1] because last string is empty after splitting
        utterances = raw_dialogues[i].strip().split('__eou__')[:-1]

        for j in range(len(utterances) - num_contexts):

            context_lst = utterances[j : j + num_contexts]
            context_str = ' '.join(context_lst)
            response_str = utterances[j + num_contexts]
            test_dict[context_str] = response_str

    return test_dict


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    test_dict = build_dd_test_dict(tokenizer)

    for context, response in test_dict.items():
        print(context, ' | ', response)
