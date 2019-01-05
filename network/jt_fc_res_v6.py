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
ln = tf.contrib.layers.layer_norm
relu = tf.nn.relu
tanh = tf.nn.tanh
lrelu = tf.nn.leaky_relu
adam = tf.train.AdamOptimizer
adamw = tf.contrib.opt.AdamWOptimizer
lstm = tf.nn.rnn_cell.LSTMCell

def block_residual(inputs, is_training, activation=relu, num_nodes=256):
#    outs = inputs

    outs = dense(inputs, num_nodes)
    outs = bn(outs, training=is_training)
    outs = activation(outs)

    resi = dense(outs, num_nodes)
    resi = bn(resi, training=is_training)
    resi = activation(resi)

    outs = outs + resi
    return outs

class Model(object):
    def __init__(self):
        self.logger = get_logger('Model')

    def get_model(self, num_classes, activation=tanh):
        len_max = opt.max_len
        size_voca = opt.unigram_hash_size + 1
        rate_dropout = opt.rate_dropout

        uni = tf.placeholder(tf.int32, shape=(None, len_max))
        w_uni = tf.placeholder(tf.float32, shape=(None, len_max))
        img_feat = tf.placeholder(tf.float32, shape=(None, opt.len_img_feat))
        price = tf.placeholder(tf.float32, shape=(None, ))
        targets = tf.placeholder(tf.float32, shape=(None, num_classes))
        is_training = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32)

        embedding = tf.get_variable('embedding', shape=(size_voca, opt.size_embedding), dtype=tf.float32)

        len_max = 16
        uni_ = tf.slice(uni, [0, 0], [-1, len_max])
        w_uni_ = tf.slice(w_uni, [0, 0], [-1, len_max])

        outs = tf.nn.embedding_lookup(embedding, uni_) # batch_size * len_max * size_embedding
        outs_w = tf.expand_dims(w_uni_, axis=2) # batch_size * len_max * 1
        outs_i = img_feat
        outs_p = tf.expand_dims(price, axis=1)

        clipped_outs_w = tf.clip_by_value(outs_w, 0.0, 1.0)

        outs = tf.multiply(outs_w, outs) # batch_size * len_max * size_embedding
#        outs = tf.concat([outs, outs_w], axis=2)

        outs = tf.reverse(outs, axis=[1])
        outs = tf.unstack(outs, len_max, axis=1)

        cell_lstm = lstm(128, name='basic_lstm_cell', forget_bias=1.0)
        outs, states = tf.nn.static_rnn(cell_lstm, outs, dtype=tf.float32)
        outs = outs[-1]

#        outs = dropout(outs, rate=rate_dropout, training=is_training)

        outs = dense(outs, 256)
        outs = bn(outs, training=is_training)
        outs = activation(outs)

        outs = dense(outs, 512)
        outs = bn(outs, training=is_training)
        outs = activation(outs)

#        outs = ln(outs)
#        outs_i = ln(outs_i)
#        outs_p = ln(outs_p)

#        for _ in range(0, 3):
#            outs = block_residual(outs, is_training, activation=activation, num_nodes=512)

        # output layer
        outs = dense(outs, num_classes)

        probs = tf.nn.softmax(outs)
        preds = tf.argmax(probs, axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=outs)
        loss = tf.reduce_mean(loss)
        opt_adam = adam(learning_rate=learning_rate)
#        opt_adam = adamw(opt.rate_weight_decay, learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = opt_adam.minimize(loss)

        model = {
            'uni': uni,
            'w_uni': w_uni,
            'img_feat': img_feat,
            'price': price,
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
