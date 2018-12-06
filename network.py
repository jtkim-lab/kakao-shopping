# -*- coding: utf-8 -*-

import tensorflow as tf

from misc import get_logger, Option
opt = Option('./config.json')

acc = tf.metrics.accuracy
dense = tf.layers.dense
conv = tf.layers.conv1d
pool = tf.layers.max_pooling1d
flatten = tf.layers.flatten
dropout = tf.layers.dropout
bn = tf.layers.batch_normalization
relu = tf.nn.relu
tanh = tf.nn.tanh


class Model(object):
    def __init__(self):
        self.logger = get_logger('Model')

    def get_model(self, num_classes, activation=tanh):
        len_max = opt.max_len
        size_voca = opt.unigram_hash_size + 1

        uni = tf.placeholder(tf.int32, shape=(None, len_max))
        w_uni = tf.placeholder(tf.float32, shape=(None, len_max))
        targets = tf.placeholder(tf.float32, shape=(None, num_classes))
        is_training = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32)

        embedding = tf.get_variable('embedding', shape=(size_voca, opt.size_embedding), dtype=tf.float32)
        outs = tf.nn.embedding_lookup(embedding, uni)
        outs_w = tf.expand_dims(w_uni, axis=1)
        
        rate_dropout = 0.4
        bias_1 = tf.get_variable('bias_1', shape=(1, opt.size_embedding), dtype=tf.float32)
        outs = tf.matmul(outs_w, outs) + bias_1
        outs = tf.squeeze(outs, axis=1)

#        outs = tf.expand_dims(outs, axis=2)

#        outs = conv(outs, 32, 3, activation=activation, padding='same')
#        outs = pool(outs, 2, 2)
#        outs = bn(outs, training=is_training)

#        outs = conv(outs, 64, 3, activation=activation, padding='same')
#        outs = pool(outs, 2, 2)
#        outs = bn(outs, training=is_training)

#        outs = flatten(outs)

        outs = dense(outs, 256)
#        outs = dropout(outs, rate=rate_dropout, training=is_training)
        outs = activation(outs)
        outs = bn(outs, training=is_training)

        outs = dense(outs, 512)
#        outs = dropout(outs, rate=rate_dropout, training=is_training)
        outs = activation(outs)
        outs = bn(outs, training=is_training)

        outs = dense(outs, 1024)
#        outs = dropout(outs, rate=rate_dropout, training=is_training)
        outs = activation(outs)
        outs = bn(outs, training=is_training)

        outs = dense(outs, num_classes)

        probs = tf.nn.softmax(outs)
        preds = tf.argmax(probs, axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=outs)
        loss = tf.reduce_mean(loss)
        opt_adam = tf.train.AdamOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = opt_adam.minimize(loss)

        model = {
            'uni': uni,
            'w_uni': w_uni,
            'targets': targets,
            'is_training': is_training,
            'learning_rate': learning_rate,
            'outs': outs,
            'probs': probs,
            'preds': preds,
            'loss': loss,
            'optimizer': optimizer,
        }
        return model


if __name__ == '__main__':
    model = Model()
    model.get_model(17)
