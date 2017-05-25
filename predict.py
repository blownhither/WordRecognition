import tensorflow as tf
import SimpleCNN
import train
from config import *

# saver = tf.train.Saver()
sess = tf.Session()

# new_saver = tf.train.import_meta_graph('model.meta')
# new_saver.restore(sess, tf.train.latest_checkpoint('./'))

test = train.read_all_specs()

SimpleCNN.run(test, test, train.SHAPE, train=False)
