# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import pickle
import importlib

import fire
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from datetime import datetime
from misc import get_logger, Option

opt = Option('./config.json')
cate1 = json.loads(open(opt.cate1, 'r').read())
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


class Classifier():
    def __init__(self):
        self.logger = get_logger('Classifier')
        self.num_classes = 0

    def get_sample_generator(self, ds, batch_size):
        left = 0
        limit = ds['uni'].shape[0]

        while True:
            right = min(left + batch_size, limit)
            X = [ds[t][left:right, :] for t in ['uni', 'w_uni']]
            Y = ds['cate'][left:right]
            yield X, Y
            left = right
            if right == limit:
                left = 0

    def get_inverted_cate1(self, cate1):
        inv_cate1 = {}
        for cur_elem in ['b', 'm', 's', 'd']:
            inv_cate1[cur_elem] = {val: key for key, val in cate1[cur_elem].items()}
        return inv_cate1

    def get_batch(self, target_data, num_data, ind_start, batch_size):
        if num_data < ind_start + batch_size:
            cur_indices = np.arange(ind_start, num_data)
        else:
            cur_indices = np.arange(ind_start, ind_start + batch_size)
        cur_uni = target_data['uni'][cur_indices, :]
        cur_w_uni = target_data['w_uni'][cur_indices, :]
        cur_img_feat = target_data['img_feat'][cur_indices, :]
        cur_price = target_data['price'][list(cur_indices)]
        cur_cate = target_data['cate'][cur_indices, :]

        return cur_uni, cur_w_uni, cur_img_feat, cur_price, cur_cate

    def write_preds(self, data, pred_y, meta, out_path, readable, str_mode=None):
        pid_order = []
        if str_mode == 'dev' or str_mode == 'test':
            if str_mode == 'dev':
                list_data = opt.dev_data_list
            elif str_mode == 'test':
                list_data = opt.test_data_list
            for path_data in list_data:
                h = h5py.File(path_data, 'r')
                h = h[str_mode]
                cur_pid = h['pid'][::]
                pid_order.extend(cur_pid)
        elif str_mode == 'train':
            pid_order = data['pid']
        else:
            for path_data in opt.dev_data_list:
                h = h5py.File(path_data, 'r')['dev']
                cur_pid = h['pid'][::]
                pid_order.extend(cur_pid)

        y2l = {i: s for s, i in meta['y_vocab'].items()}
        y2l = list(map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0])))
        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}
        for pid, p in zip(data['pid'], pred_y):
            pid = pid.decode('utf-8')
            y = p
            label = y2l[y]
            tkns = list(map(int, label.split('>')))
            b, m, s, d = tkns
            assert b in inv_cate1['b']
            assert m in inv_cate1['m']
            assert s in inv_cate1['s']
            assert d in inv_cate1['d']
            tpl = '{pid}\t{b}\t{m}\t{s}\t{d}'
            if readable:
                b = inv_cate1['b'][b]
                m = inv_cate1['m'][m]
                s = inv_cate1['s'][s]
                d = inv_cate1['d'][d]
            rets[pid] = tpl.format(pid=pid, b=b, m=m, s=s, d=d)
        no_answer = '{pid}\t-1\t-1\t-1\t-1'
        with open(out_path, 'w') as fout:
            for pid in pid_order:
                pid = pid.decode('utf-8')
                ans = rets.get(pid, no_answer.format(pid=pid))
                fout.write(ans)
                fout.write('\n')

    def predict(self, path_root, model_root, str_test, div_test, path_out, readable=False):
        path_meta = os.path.join(path_root, 'meta')
        meta = pickle.loads(open(path_meta, 'rb').read())
        self.logger.info('# of classes in train {}'.format(len(meta['y_vocab'])))
        self.num_classes = len(meta['y_vocab'])

        path_test = os.path.join(str_test, 'data.h5py')
        data_test = h5py.File(path_test, 'r')
        data_test = data_test[div_test]
        
        num_samples_test = data_test['uni'].shape[0]
        batch_size = opt.batch_size
        self.logger.info('# of test samples {}'.format(num_samples_test))

        preds_test = None
        obj_model = importlib.import_module('.{}'.format(opt.network), package='network').Model()
        model = obj_model.get_model(self.num_classes)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            path_checkpoint = tf.train.latest_checkpoint(opt.path_model)
            self.logger.info('load {}'.format(path_checkpoint))
            saver.restore(sess, path_checkpoint)

            iter_total = int(num_samples_test / batch_size)
            if num_samples_test % batch_size > 0:
                iter_total += 1
            for ind_iter in range(0, iter_total):
                uni_test, w_uni_test, img_feat_test, price_test, targets_test = self.get_batch(data_test, num_samples_test, ind_iter * batch_size, batch_size)

                cur_preds = sess.run(model['preds'], {
                    model['uni']: uni_test,
                    model['w_uni']: w_uni_test,
                    model['img_feat']: img_feat_test,
                    model['price']: price_test,
                    model['is_training']: False,
                })
                if preds_test is None:
                    preds_test = cur_preds
                else:
                    preds_test = np.concatenate((preds_test, cur_preds))

        # TODO(Jungtaek): make it smarter
        if 'train' in str_test:
            str_mode = 'train'
        elif 'dev' in str_test:
            str_mode = 'dev'
        elif 'test' in str_test:
            str_mode = 'test'
        else:
            str_mode = None
        self.write_preds(data_test, preds_test, meta, path_out, readable, str_mode=str_mode)

    def train(self, path_root, is_scratch=True):
        path_data = os.path.join(path_root, 'data.h5py')
        path_meta = os.path.join(path_root, 'meta')
        data = h5py.File(path_data, 'r')
        meta = pickle.loads(open(path_meta, 'rb').read())

        if not os.path.exists(opt.path_model):
            os.makedirs(path_model)

        self.num_classes = len(meta['y_vocab'])

        data_train = data['train'] # ['cate', 'pid', 'uni', 'w_uni'] + ['price', 'img_feat']
        data_dev = data['dev']

        num_samples_train = data_train['uni'].shape[0]
        num_samples_dev = data_dev['uni'].shape[0]
        batch_size = opt.batch_size

        self.logger.info('train cate {} pid {} uni {} w_uni {} price {} img_feat {}'.format(data_train['cate'].shape, data_train['pid'].shape, data_train['uni'].shape, data_train['w_uni'].shape, data_train['price'], data_train['img_feat']))
        self.logger.info('dev cate {} pid {} uni {} w_uni {} price {} img_feat {}'.format(data_dev['cate'].shape, data_dev['pid'].shape, data_dev['uni'].shape, data_dev['w_uni'].shape, data_dev['price'], data_dev['img_feat']))
        self.logger.info('# of classes %s' % len(meta['y_vocab']))

        self.logger.info('# of train samples %s' % data_train['cate'].shape[0])
        self.logger.info('# of dev samples %s' % data_dev['cate'].shape[0])

        obj_model = importlib.import_module('.{}'.format(opt.network), package='network').Model()
        model = obj_model.get_model(self.num_classes)
        iter_total = tf.Variable(0, tf.int32)
        add_iter = tf.assign_add(iter_total, 1)

        uni_dev, w_uni_dev, img_feat_dev, price_dev, targets_dev = self.get_batch(data_train, num_samples_train, 0, int(num_samples_dev / 100))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # init
            if is_scratch:
                sess.run(tf.global_variables_initializer())
            else:
                path_checkpoint = tf.train.latest_checkpoint(opt.path_model)
                self.logger.info('load {}'.format(path_checkpoint))
                saver.restore(sess, path_checkpoint)

            #######################
            
            # summary writer
            if not opt.name_tensorboard or opt.name_tensorboard.isspace():
                event_dir = os.path.join(opt.path_tensorboard, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            else:
                event_dir = os.path.join(opt.path_tensorboard, datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') + opt.name_tensorboard)
            
            if not os.path.isdir(event_dir):
                os.mkdir(event_dir)

            summary_writer = tf.summary.FileWriter(event_dir)

            #######################

            # projector
            # code is from https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector

            # label
            metadata = os.path.join(event_dir, 'metadata.tsv')

            with open(metadata, 'w') as metadata_file:
                for row in targets_dev:
                    metadata_file.write('%d\n' % row)

            emb_var = tf.Variable(img_feat_dev, name='embeddings')

            projector_saver = tf.train.Saver([emb_var])

            sess.run(emb_var.initializer)
            projector_saver.save(sess, os.path.join(event_dir, 'img_feat.ckpt'))

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = emb_var.name
            embedding.metadata_path = metadata

            # visualise the embeddings
            projector.visualize_embeddings(summary_writer, config)

            #######################

            for ind_epoch in range(0, opt.num_epochs):
                self.logger.info('current epoch {}'.format(ind_epoch + 1))
                for ind_iter in range(0, int(num_samples_train / batch_size)):
                    uni_train, w_uni_train, img_feat_train, price_train, targets_train = self.get_batch(data_train, num_samples_train, ind_iter * batch_size, batch_size)
                    cur_lr = opt.lr * opt.rate_decay**int(sess.run(iter_total) / float(opt.step_decay))
                    _, cur_loss, _, cur_iter = sess.run([model['optimizer'], model['loss'], add_iter, iter_total], {
                        model['uni']: uni_train,
                        model['w_uni']: w_uni_train,
                        model['img_feat']: img_feat_train,
                        model['price']: price_train,
                        model['targets']: targets_train,
                        model['is_training']: True,
                        model['learning_rate']: cur_lr,
                    })

                    if cur_iter % opt.step_display == 0:
                        self.logger.info('cur_iter {} cur_loss {:.4f} cur_lr {:.2E}'.format(cur_iter, cur_loss, cur_lr))
                        cur_loss_dev = sess.run(model['loss'], {
                            model['uni']: uni_dev,
                            model['w_uni']: w_uni_dev,
                            model['img_feat']: img_feat_dev,
                            model['price']: price_dev,
                            model['targets']: targets_dev,
                            model['is_training']: False,
                        })
                        self.logger.info('cur_loss_dev {:.4f}'.format(cur_loss_dev))
                        
                        train_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss/train', simple_value=cur_loss)])
                        eval_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss/eval', simple_value=cur_loss_dev)])
                        lr_summary = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=cur_lr)])
                        
                        summary_writer.add_summary(train_loss_summary, global_step=cur_iter)
                        summary_writer.add_summary(eval_loss_summary, global_step=cur_iter)
                        summary_writer.add_summary(lr_summary, global_step=cur_iter)
	    

                if (ind_epoch + 1) % opt.step_save == 0:
                    saver.save(sess, os.path.join(opt.path_model, opt.str_model), global_step=cur_iter)


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train, 'predict': clsf.predict})
