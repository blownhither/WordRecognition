import tensorflow as tf
import SimpleCNN
import app
from config import *

# saver = tf.train.Saver()
sess = tf.Session()

# new_saver = tf.train.import_meta_graph('model.meta')
# new_saver.restore(sess, tf.train.latest_checkpoint('./'))

train, test = app.read_all_specs().split(0.1)

SimpleCNN.run(train, test, app.SHAPE, train=False)
