import numpy as np

from config import *


SHAPE = np.fromfile('spec/config.txt', sep=' ').astype(np.int)
LINE_LEN = SHAPE[0] * SHAPE[1]
N_TYPES = len(words)


def read_spec(path):
    data = np.fromfile(path, sep=' ')
    return data.reshape([-1, LINE_LEN])    # to line shape


def read_all_specs():
    x = np.zeros((1, LINE_LEN))
    y = np.zeros((1, N_TYPES))
    for i, v in enumerate(words):
        data = read_spec('./spec/' + v + '.txt')
        x = np.concatenate([x, data], axis=0)
        temp = np.zeros((1, N_TYPES))
        temp[0][i] = 1
        y = np.concatenate([y, temp.repeat(data.shape[0], axis=0)], axis=0)

        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == LINE_LEN
        assert y.shape[1] == N_TYPES
    x = x[1:]
    y = y[1:]
    return x, y


def main():
    data = read_all_specs()
    print(data)


if __name__ == '__main__':
    main()
