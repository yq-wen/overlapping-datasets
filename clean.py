if __name__ == '__main__':

    train_f = open('data/ijcnlp_dailydialog/train/dialogues_train.txt', mode='r')
    test_f = open('data/ijcnlp_dailydialog/test/dialogues_test.txt', mode='r')
    valid_f = open('data/ijcnlp_dailydialog/validation/dialogues_validation.txt', mode='r')

    train_lines = set(train_f.readlines())

    with open('dialogues_test_clean.txt', mode='w') as f:
        for line in test_f:
            if line in train_lines:
                print('skipping:', line)
            else:
                f.write(line)

    with open('dialogues_validation_clean.txt', mode='w') as f:
        for line in valid_f:
            if line in train_lines:
                print('skipping:', line)
            else:
                f.write(line)
