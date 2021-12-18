import torch
import pandas as pd

from pathlib import PosixPath
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead


class DailyDialogueDataset(Dataset):

    def __init__(self, tokenizer, max_length=64, path=None):

        df = pd.read_csv(path)

        if 'gpt2' in tokenizer.name_or_path or 'DialoGPT' in tokenizer.name_or_path:

            contexts = []
            responses = []

            samples = []

            for index, row in df.iterrows():
                context = row['context'].strip().lower()
                response = row['response'].strip().lower()
                sample = '{} {} {} {}'.format(context, tokenizer.sep_token, response, tokenizer.eos_token)
                samples.append(sample)

            encoded = tokenizer(
                samples,
                max_length=max_length * 2,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )

            self.data = dict()
            self.data['input_ids'] = encoded['input_ids']
            self.data['attention_mask'] = encoded['attention_mask']
            self.data['labels'] = encoded['input_ids'].clone()
            self.data['labels'][encoded['attention_mask'] == 0] = -100

        else:

            contexts = []
            responses = []

            for index, row in df.iterrows():
                contexts.append(row['context'].lower())
                responses.append(row['response'].lower())

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
