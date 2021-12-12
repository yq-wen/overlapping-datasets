from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd

QUANTILES = [0.8, 0.9, 0.95, 0.975, 0.99]

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    df = pd.read_csv('test.txt', sep='\t', header=None)
    df = df.rename(columns={0: 'context', 1: 'response'})
    # df = pd.read_csv('data/alpha_1.0_dailydialog/utterances/test.csv')
    df['context'].apply(lambda x: tokenizer(x).input_ids)
    lengths = df['context'].apply(lambda x: len(tokenizer(x).input_ids))
    print(lengths.quantile(QUANTILES))
