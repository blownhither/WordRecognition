import numpy as np


def hamming_window(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


def hanning_window(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))


def mel_frequency(f):
    return 2595 * np.log(1 + np.array(f, dtype=np.float) / 700.0)


def spectrum(x):
    pass