import argparse
from path import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", metavar='DIR', help='path to original dataset')

    args = parser.parse_args()

    sample_data_path = Path(args.dataset_dir)/"RPKM.reversed.99.txt"
    train_data_path = Path(args.dataset_dir)/"train.txt"
    val_data_path = Path(args.dataset_dir)/"val.txt"

    with open(sample_data_path) as f:
        train = []
        val = []

        for line in f:
            if np.random.rand() > .3:
                train.append(line)
            else:
                val.append(line)

        with open(train_data_path, "w+") as t:
            for t_line in train:
                t.write(t_line)

        with open(val_data_path, 'w+') as v:
            for v_line in val:
                v.write(v_line)

        print("******************** Data Preparation Finished *************************")
        print("The number of items in training set is {}".format(len(train)))
        print("The number of items in validation set is {}".format(len(val)))


if __name__ == '__main__':
    main()
