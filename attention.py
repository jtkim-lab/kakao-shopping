import tensorflow as tf
import numpy as np

relu = tf.nn.relu
dense = tf.layers.dense
softplus = tf.nn.softplus
softmax = tf.nn.softmax
layer_norm = tf.contrib.layers.layer_norm

# X: batch_size * N * D
# Y: batch_size * M * D
def MAB(X, Y, n_units, n_heads):
    Q = dense(X, n_units, activation=relu)
    K = dense(Y, n_units, activation=relu)
    V = dense(Y, n_units, activation=relu)

    Q_ = tf.concat(tf.split(Q, n_heads, 2), 0)
    K_ = tf.concat(tf.split(K, n_heads, 2), 0)
    V_ = tf.concat(tf.split(V, n_heads, 2), 0)

    att = softmax(tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) / tf.sqrt(tf.cast(n_units, tf.float32)))
    outs = tf.matmul(att, V_)
    outs = tf.concat(tf.split(outs, n_heads, 0), 2)
    outs += Q
    outs = layer_norm(outs, begin_norm_axis=-1)
    outs = layer_norm(outs + dense(outs, n_units, activation=relu), begin_norm_axis=-1)

    return outs

def IMAB(X, n_inds, n_units, n_heads, var_name='ind'):
    I = tf.get_variable(var_name, shape=[1, n_inds, n_units])
    I = tf.tile(I, [tf.shape(X)[0], 1, 1])
    return MAB(I, X, n_units, n_heads)
