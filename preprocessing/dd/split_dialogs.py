import argparse
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../../data/proposed/dialogs/all_dialogs.txt')
    parser.add_argument('--num-test', default=1000)
    parser.add_argument('--num-valid', default=1000)

    args = parser.parse_args()

    with open(args.path, mode='r') as f:
        dialogs = f.readlines()

    random.seed(0)
    random.shuffle(dialogs)

    test = dialogs[:args.num_test]
    valid = dialogs[args.num_test:args.num_test+args.num_valid]
    train = dialogs[args.num_test+args.num_valid:]

    with open('train_dialogs.txt', mode='w') as f:
        f.writelines(train)
    with open('valid_dialogs.txt', mode='w') as f:
        f.writelines(valid)
    with open('test_dialogs.txt', mode='w') as f:
        f.writelines(test)

    print('done')
