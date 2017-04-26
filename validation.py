# coding=UTF-8
import wave
import os

from config import *

PREFIX = 'data/'

import app
b = app.read_all_specs()
b.split(test_ratio=0.05)
