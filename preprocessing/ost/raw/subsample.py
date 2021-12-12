import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--num-samples', type=int)

    args = parser.parse_args()

    with open(args.path, mode='r') as f_in:
        with open('samples.txt', mode='w') as f_out:
            f_out.write('context\tresponse\n')
            for i, line in enumerate(f_in):
                f_out.write(line[2:])
                if i == args.num_samples - 1:
                    break
