# -*- coding: utf-8 -*-

import tensorflow as tf

from attention import MAB, IMAB
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
lrelu = tf.nn.leaky_relu


class Model(object):
    def __init__(self):
        self.logger = get_logger('Model')

    def get_model(self, num_classes, activation=relu):
        len_max = opt.max_len
        size_voca = opt.unigram_hash_size + 1
        rate_dropout = 0.5

        uni = tf.placeholder(tf.int32, shape=(None, len_max))
        w_uni = tf.placeholder(tf.float32, shape=(None, len_max))
        targets = tf.placeholder(tf.float32, shape=(None, num_classes))
        is_training = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32)

        embedding = tf.get_variable('embedding', shape=(size_voca, opt.size_embedding), dtype=tf.float32)
        outs = tf.nn.embedding_lookup(embedding, uni) # batch_size * len_max * size_embedding
        outs_w = tf.expand_dims(w_uni, axis=2) # batch_size * len_max * 1
        outs_w = tf.tile(outs_w, [1, 1, opt.size_embedding]) # batch_size * len_max * size_embedding
        outs += outs_w
        outs_raw = outs
        outs = dropout(outs, rate=rate_dropout, training=is_training)

        # encoder
#        outs = MAB(outs, outs, 128, 4)
        outs_enc = MAB(outs, outs, 128, 4)

        outs = dropout(outs_raw, rate=rate_dropout, training=is_training)

        # decoder
        for cur_ind in range(0, 1):
            outs = IMAB(outs, 128, 128, 4, var_name='seed{}'.format(cur_ind))
            outs = MAB(outs, outs_enc, 128, 4)

        outs = dense(outs, 1)
#        outs = dropout(outs, rate=rate_dropout, training=is_training)
#        outs = activation(outs)
        outs = tf.squeeze(outs, axis=2)

        outs = dense(outs, 256)
#        outs = dropout(outs, rate=rate_dropout, training=is_training)
        outs = bn(outs, training=is_training)
        outs = activation(outs)

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
