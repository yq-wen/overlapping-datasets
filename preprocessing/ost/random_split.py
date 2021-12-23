import random
import sys
sys.path.append('../../dd')

from pathlib import PosixPath
from flatten import flatten

SEPARATOR = ' __eou__ '

def get_dialogs(file):

    dialogs = []

    with open(file, mode='r') as f:

        utterances = []

        for line in f:

            seq_num, content = line.split(' ', 1)
            content = content.strip()

            if seq_num == '1':
                if utterances:
                    dialogs.append(SEPARATOR.join(utterances) + SEPARATOR)
                    utterances = []

            if '\t' in content:
                ut_1, ut_2 = content.split('\t')
                utterances.append(ut_1)
                utterances.append(ut_2)
            else:
                utterances.append(content)

        if utterances:
            # Add an extra __eou__ to match the format in
            # DailyDialog
            dialogs.append(SEPARATOR.join(utterances) + SEPARATOR)
            utterances = []

    return dialogs

def dump_split(files, output_path, sample=None):

    dialogs = []

    for file in files:
        dialogs += get_dialogs(file)

    if sample:
        dialogs = random.sample(dialogs, sample)

    with open(output_path, mode='w') as f:
        f.write('\n'.join(dialogs))

if __name__ == '__main__':

    NUM_TEST  = 1500
    NUM_VALID = 1500
    NUM_TRAIN = 1500

    input_dir = PosixPath('dedup_samples')
    files = list(input_dir.glob('*.txt'))

    random.seed(0)
    random.shuffle(files)

    dump_split(files[0:NUM_TEST], 'test.txt', sample=2000)
    dump_split(files[NUM_TEST:NUM_TEST+NUM_VALID], 'valid.txt', sample=2000)
    dump_split(files[NUM_TEST+NUM_VALID:NUM_TEST+NUM_VALID+NUM_TRAIN], 'train.txt')

    flatten('test.txt').to_csv('dirty_test.csv', index=False)
    flatten('valid.txt').to_csv('dirty_valid.csv', index=False)
    flatten('train.txt').to_csv('dirty_train.csv', index=False)

    print('done')
