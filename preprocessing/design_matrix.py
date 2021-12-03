import torch
import pandas as pd
import preprocess_utils

from torch.utils.data import Dataset, DataLoader


class DesignMatrix(Dataset):

    def __init__(self, w2i, df, n=1):
        '''
        Arguments:
            w2i (dict): mapping from grams to indices
            df (Dataframe)
            n (int): order of n-grams
        '''
        self.w2i = w2i
        self.df = df
        self.vocab_size = len(w2i)
        self.n = n

    def __getitem__(self, index):

        context = self.df['context'][index]
        grams = preprocess_utils.get_grams(context, n=self.n)
        context_bow = torch.zeros(self.vocab_size, dtype=torch.float16)
        for gram in grams:
            context_bow[self.w2i[gram]] = 1

        response = self.df['response'][index]
        grams = preprocess_utils.get_grams(response, n=self.n)
        response_bow = torch.zeros(self.vocab_size, dtype=torch.float16)
        for gram in grams:
            response_bow[self.w2i[gram]] = 1

        return {
            'context_bow': context_bow,
            'response_bow': response_bow
        }

    def __len__(self):
        return self.df.shape[0]
