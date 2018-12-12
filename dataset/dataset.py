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

        s_array = np.array(list(sample.get('X')))
        s_array = s_array.view(dtype=np.int16, type=np.matrix)
        s_array = s_array.view(-1, 24)
        s_array = s_array.astype(float)

        y_float = float(self.samples[index].get('y'))
        return s_array, y_float

    def __len__(self):
        return len(self.samples)
