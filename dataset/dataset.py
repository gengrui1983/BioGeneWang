import array
import random

import numpy as np
import torch
import torch.utils.data as data
from path import Path


class BioData(data.Dataset):

    """
    The dataset should be the format of following:
    y: 0.233
    X: (If there are A, B, C and D four elements)
        origin - ABBCD
        formatted -
            1 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 1
            A------|B------|B------|C------|D-----|
    """
    def __init__(self, root, seed=None, train=True):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.samples = []
        self.is_train = train

        self.parse_data()

    def parse_data(self):

        dat = []
        value = []
        file_name = Path(self.root)/'train.txt' if self.is_train else 'val.txt'
        with open(file_name) as f:
            for line in f:
                xy = line.split()
                value.append(xy[0])
                dat.append(xy[1])

        for d, y in zip(dat, value):
            sample = {'y': y, 'X': d}
            self.samples.append(sample)

        random.shuffle(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # print("sample:{}\nThe length of sample is:{}".format(sample, len(sample.get('X'))/4))

        s_array = np.array(list(sample.get('X')))
        # print("1, ", s_array)
        # print(s_array)
        s_array = list(map(int, s_array))
        s_array = torch.Tensor(s_array)
        s_array = s_array.view(-1, 4)
        s_array.unsqueeze_(0)

        # print("array:", s_array)
        # print("array.dtype", s_array.dtype)
        # print("array.size", s_array.size())

        y_float = float(self.samples[index].get('y'))
        return s_array, y_float

    def __len__(self):
        return len(self.samples)
