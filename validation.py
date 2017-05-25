# coding=UTF-8
import wave
import os
from matplotlib import pyplot as plt

from config import *
import train

PREFIX = 'data/'


b = train.read_all_specs()
print(b.y_distribution())

# b.split(test_ratio=0.05)
y_dist = [167., 227., 203., 197., 163., 228., 167., 208., 144.,
          220., 254., 164., 145., 175., 169., 302., 227., 372.,
          375., 366., 363., 324., 364., 373., 387., 284., 247.,
          345., 364., 367.]


plt.bar(range(len(y_dist)), y_dist, tick_label=words)

plt.show()