import torch

from pathlib import PosixPath
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead


class DailyDialogueDataset(Dataset):

    def __init__(self, tokenizer, split='train', num_contexts=1, max_length=20, dir='data/clean_dailydialog'):

        assert split in ['train', 'validation', 'test']

        dialogues = 'dialogues_{}_clean.txt'.format(split)


        with open(PosixPath(dir, split, dialogues), mode='r') as f_dialogues:
            raw_dialogues = f_dialogues.readlines()
        num_dialogues = len(raw_dialogues)

        contexts = []
        responses = []

        for i in range(num_dialogues):

            # [:-1] because last string is empty after splitting
            utterances = raw_dialogues[i].strip().split('__eou__')[:-1]

            for j in range(len(utterances) - num_contexts):

                context_lst = utterances[j : j + num_contexts]
                context_str = '<sep>'.join(context_lst)

                contexts.append(context_str)
                responses.append(utterances[j + num_contexts] + ' ' + tokenizer.eos_token)

        assert len(contexts) == len(responses)

        inputs = tokenizer(
            contexts,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        labels = tokenizer(
            responses,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        labels['input_ids'][labels['attention_mask'] == 0] = -100

        self.data = dict()
        self.data['input_ids'] = inputs['input_ids']
        self.data['attention_mask'] = inputs['attention_mask']
        self.data['labels'] = labels['input_ids']


    def __getitem__(self, index):
        return {
            'input_ids': self.data['input_ids'][index],
            'attention_mask': self.data['attention_mask'][index],
            'labels': self.data['labels'][index],
            'indices': index,
        }

    def __len__(self):
        return len(self.data['input_ids'])


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # model = AutoModelWithLMHead.from_pretrained("t5-base")

    dataset = DailyDialogueDataset(tokenizer)
