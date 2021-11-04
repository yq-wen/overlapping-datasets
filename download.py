import argparse
from transformers import AutoTokenizer, AutoModelWithLMHead

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Sciprt for downloading pretrained LMs')

    parser.add_argument('--model', type=str, default='t5-base')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelWithLMHead.from_pretrained(args.model)

    tokenizer.save_pretrained(args.model)
    model.save_pretrained(args.model)
