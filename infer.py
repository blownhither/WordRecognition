# encoding=UTF-8

import tensorflow as tf
import numpy as np

import config

from Util.BatchMaker import BatchMaker

# routines
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    """
    :param x:
    :param w: [n_rows, n_cols, n_layers, n_kernels]
    :return:
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    # strides is the offset of window in each step


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # here ksize and strides are in [batch, height, width, channels]


# main function
def infer(predict_model_path=""):
    sess = tf.InteractiveSession()

    n_outputs = 30
    n_rows, n_cols = (32, 78)
    line_len = n_rows * n_cols

    x = tf.placeholder(tf.float32, [None, line_len])
    y_ = tf.placeholder(tf.float32, [None, n_outputs])
    x_image = tf.reshape(x, [-1, n_rows, n_cols, 1])

    L1_KERNEL = 32
    L2_KERNEL = 64
    FC_NODE = 2048

    # the 1st convolution layer
    w_conv1 = weight_variable([5, 5, 1, L1_KERNEL])    # 32 conv kernel, each is 5rows * 5cols * 1channel
    b_conv1 = bias_variable([L1_KERNEL])
    # the virtue of ReLu activation function is its not being susceptible to gradient diffusion
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)        # relu(f) === max(f, 0)
    h_pool1 = max_pool_2x2(h_conv1)

    # the 2nd convolution layer
    w_conv2 = weight_variable([5, 5, L1_KERNEL, L2_KERNEL])   # full connection between two conv layers
    b_conv2 = bias_variable([L2_KERNEL])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # now the shape of tensor is 64kernels * [7*7]image

    ignore, n_rows_, n_cols_, n_kernels = h_pool2.get_shape().as_list()

    # reshape and full connection layer
    w_fc1 = weight_variable([n_rows_ * n_cols_ * L2_KERNEL, FC_NODE])             # 1024 is arbitrary
    b_fc1 = bias_variable([FC_NODE])
    h_pool2_flat = tf.reshape(h_pool2, [-1, n_rows_ * n_cols_ * L2_KERNEL])    # flatten for kx+b
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # Dropout Layer
    keep_prob = tf.placeholder(tf.float32)                  # threshold passed at runtime
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)            # drop some to avoid over-fitting

    # Softmax layer
    w_fc2 = weight_variable([FC_NODE, n_outputs])
    b_fc2 = bias_variable([n_outputs])
    y_fc2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    y_conv = tf.nn.softmax(y_fc2)

    saver = tf.train.Saver()
    saver.restore(sess, predict_model_path)

    return y_conv, (x, y_, keep_prob)


def predict(y_conv, tf_vars, data):
    pred = y_conv.eval(feed_dict={
        tf_vars[0]: data,
        tf_vars[1]: np.zeros((1, 30)),
        tf_vars[2]: 1.0
    })
    return pred
