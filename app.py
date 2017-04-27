import numpy as np
import tensorflow as tf

from config import *
from Util.BatchMaker import BatchMaker
import SimpleCNN

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
        temp[0][i] = 1                      # one hot representation
        y = np.concatenate([y, temp.repeat(data.shape[0], axis=0)], axis=0)

        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == LINE_LEN
        assert y.shape[1] == N_TYPES
    x = x[1:]
    y = y[1:]

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    print(x.shape)
    count = 0
    for i in range(x.shape[0] - 1, -1, -1):
        if x[i].max() - x[i].min() < 0.1:
            count += 1
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    print(count)
    return BatchMaker(x, y)


def main():
    b = read_all_specs()
    train, test = b.split(test_ratio=0.05)
    print("Loaded data train %s, test %s" % (str(train.shape()), str(test.shape())))

    SimpleCNN.run(train, test, SHAPE)
    # saver = tf.train.Saver()
    # saver.save(sess, 'model/model')

if __name__ == '__main__':
    main()
