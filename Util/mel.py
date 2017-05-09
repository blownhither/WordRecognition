import numpy as np
from scipy.fftpack import dct
from matplotlib import pyplot as plt, cm
import wave

from fft import FFT


_hamming_memo = {}
def hamming_window(N):
    if N not in _hamming_memo:
        _hamming_memo[N] = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    return _hamming_memo[N]


def hanning_window(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))


def mel_frequency(f):
    return 1125 * np.log(1 + np.array(f, dtype=np.float) / 700.0)


def imel_frequency(m):
    return 700 * (np.exp(np.array(m, dtype=np.float) / 1125.0) - 1)


_mel_bins_memo = {}
def mel_bins(nbins):
    if nbins not in _mel_bins_memo:
        mel = mel_frequency(np.linspace(300, 8000, nbins + 2))
        _mel_bins_memo[nbins] = imel_frequency(mel)
    return _mel_bins_memo[nbins]


def spectrum(x,  sample_rate=8192, win_len=128):
    assert bin(win_len).count('1') == 1
    assert len(x) >= win_len

    def _next():
        for _p in range(0, len(x) - win_len + 1, win_len//2):
            yield x[_p:_p + win_len]
        raise StopIteration()

    h = hamming_window(win_len)
    f = FFT()
    fbins = mel_bins(24)
    ibins = np.floor((win_len + 1) * fbins / sample_rate).astype(np.int)

    ret = []
    for win in _next():
        s = f.fft(win * h)
        p = np.square(np.abs(s)) / win_len
        temp = [None] * (len(ibins) - 2)
        for i in range(len(ibins) - 2):
            a, b, c = ibins[i], ibins[i+1], ibins[i+2]
            left = np.sum(p[a:b] * np.linspace(0, 1, b - a))
            right = np.sum(p[b:c] * np.linspace(1, 0, c - b))
            temp[i] = np.log(left + right + 1e-100)

        ret.append(dct(np.array(temp, dtype=np.float)))

    return np.array(ret)


def test():
    # x = np.ones(1024)
    wav = wave.open('/Users/bh/projects/signal/speech_recog/rec/14307130033-China-04.dat', 'rb')
    str_data = wav.readframes(wav.getnframes())
    x = np.fromstring(str_data, dtype=np.short)
    x = x.reshape((-1, 2)).T
    ret = spectrum(x[0])
    print(ret)
    print(ret.shape)

    ret = (ret - ret.min()) / (ret.max() - ret.min())
    plt.imshow(ret, cmap=cm.hot)
    plt.show()


if __name__ == '__main__':
    test()


