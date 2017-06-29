# encoding=UTF-8

import time
import os
import numpy as np

import infer
import config


def file_modify_time(stime):            # formatted string
    return time.strftime("%H:%M:%S\t", time.localtime(stime))


def file_modify_second(filename):       # seconds from 1970
    return os.path.getmtime(filename)


def main():
    filename = "working/working.txt"
    last = 0
    y_conv, tf_vars = infer.infer("model/model2017-06-02/model")
    while True:
        time.sleep(1)
        temp = file_modify_second(filename)
        data = np.fromfile(filename, sep=' ').reshape((1, 2496))
        if last != temp:                # modified
            print(file_modify_time(temp))

            tic = time.time()
            index = infer.predict(y_conv, tf_vars, data)
            toc = time.time()

            print("Recognized: " + config.words[index] + "\t\tTime usage: " + str(toc - tic))
            last = temp


if __name__ == '__main__':
    main()