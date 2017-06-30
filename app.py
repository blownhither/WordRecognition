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
    last = file_modify_second(filename)
    y_conv, tf_vars = infer.infer("model/model2017-06-02/model")
    while True:
        time.sleep(1)
        temp = file_modify_second(filename)
        data = np.fromfile(filename, sep=' ').reshape((1, 2496))

        # data = (data + 1.67) / (30.79  + 1.67)
        data = (data - data.min()) / (data.max() - data.min())
        if last != temp:                # modified
            print(file_modify_time(temp))

            tic = time.time()
            pred = infer.predict(y_conv, tf_vars, data)
            toc = time.time()

            index = np.argmax(pred)
            print("Recognized: " + config.words[index] + "\t\tTime usage: " + str(toc - tic))
            print("\tConfidence: %g" % pred[0, index])
            last = temp


if __name__ == '__main__':
    main()