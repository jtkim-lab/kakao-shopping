# -*- coding: utf-8 -*-

import tensorflow as tf

from misc import get_logger, Option
opt = Option('./config.json')

acc = tf.metrics.accuracy
dense = tf.layers.dense
dropout = tf.layers.dropout
relu = tf.nn.relu


class Model(object):
    def __init__(self):
        self.logger = get_logger('Model')

    def get_model(self, num_classes, activation=relu):
        max_len = opt.max_len
        voca_size = opt.unigram_hash_size + 1

        uni = tf.placeholder(tf.float32, shape=(None, max_len))
        w_uni = tf.placeholder(tf.float32, shape=(None, max_len))
        targets = tf.placeholder(tf.float32, shape=(None, num_classes))
        is_training = tf.placeholder(tf.bool)

        outs = tf.expand_dims(uni, axis=2)
        outs = dense(outs, 128)
        outs_w = tf.expand_dims(w_uni, axis=1)
        
        outs = tf.matmul(outs_w, outs)
        outs = tf.squeeze(outs, axis=1)
        outs = dense(outs, 256)
        outs = relu(outs)
        outs = dropout(outs, rate=0.5, training=is_training)
        outs = dense(outs, num_classes)

        preds = tf.argmax(tf.nn.softmax(outs), axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=outs)
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(opt.lr)
        optimizer = optimizer.minimize(loss)

        model = {
            'uni': uni,
            'w_uni': w_uni,
            'targets': targets,
            'is_training': is_training,
            'outs': outs,
            'preds': preds,
            'loss': loss,
            'optimizer': optimizer
        }
        return model


if __name__ == '__main__':
    model = Model()
    model.get_model(17)
