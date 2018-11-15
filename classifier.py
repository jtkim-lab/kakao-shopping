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

import fire
import h5py
import numpy as np
import tensorflow as tf

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from misc import get_logger, Option
from network import TextOnly, top1_acc
from network_ import Model, acc

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
        cur_indices = np.arange(ind_start, ind_start + batch_size)
        cur_uni = target_data['uni'][cur_indices, :]
        cur_w_uni = target_data['w_uni'][cur_indices, :]
        cur_cate = target_data['cate'][cur_indices, :]
        return cur_uni, cur_w_uni, cur_cate

    def write_preds(self, data, pred_y, meta, out_path, readable):
        pid_order = []
        for path_data in opt.dev_data_list:
            h = h5py.File(path_data, 'r')['dev']
            pid_order.extend(h['pid'][::])

        y2l = {i: s for s, i in meta['y_vocab'].items()}
        y2l = list(map(lambda x: x[1], sorted(y2l.items(), key=lambda x: x[0])))
        inv_cate1 = self.get_inverted_cate1(cate1)
        rets = {}
        for pid, p in zip(data['pid'], pred_y):
            pid = pid.decode('utf-8')
            y = np.argmax(p)
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

    def predict(self, path_root, model_root, test_root, test_div, out_path, readable=False):
        path_meta = os.path.join(path_root, 'meta')
        meta = pickle.loads(open(path_meta, 'rb').read())

        model_fname = os.path.join(model_root, 'model.h5')
        self.logger.info('# of classes in train %s' % len(meta['y_vocab']))
        model = load_model(
            model_fname,
            custom_objects={'top1_acc': top1_acc}
        )

        test_path = os.path.join(test_root, 'data.h5py')
        test_data = h5py.File(test_path, 'r')

        test = test_data[test_div]
        test_gen = self.get_sample_generator(test, opt.batch_size)
        total_test_samples = test['uni'].shape[0]
        steps = int(np.ceil(total_test_samples / float(opt.batch_size)))
        pred_y = model.predict_generator(
            test_gen,
            steps=steps,
            workers=opt.num_predict_workers,
            verbose=1
        )
        self.write_preds(test, pred_y, meta, out_path, readable)

    def train(self, path_root, path_out):
        path_data = os.path.join(path_root, 'data.h5py')
        path_meta = os.path.join(path_root, 'meta')
        data = h5py.File(path_data, 'r')
        meta = pickle.loads(open(path_meta, 'rb').read())

        self.weight_fname = os.path.join(path_out, 'weights')
        self.model_fname = os.path.join(path_out, 'model')
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        self.num_classes = len(meta['y_vocab'])

        data_train = data['train'] # ['cate', 'pid', 'uni', 'w_uni']
        data_dev = data['dev']

        num_samples_train = data_train['uni'].shape[0]
        num_samples_dev = data_dev['uni'].shape[0]

        self.logger.info('train cate {} pid {} uni {} w_uni {}'.format(data_train['cate'].shape, data_train['pid'].shape, data_train['uni'].shape, data_train['w_uni'].shape))
        self.logger.info('dev cate {} pid {} uni {} w_uni {}'.format(data_dev['cate'].shape, data_dev['pid'].shape, data_dev['uni'].shape, data_dev['w_uni'].shape))
        self.logger.info('# of classes %s' % len(meta['y_vocab']))

        self.logger.info('# of train samples %s' % data_train['cate'].shape[0])
        self.logger.info('# of dev samples %s' % data_dev['cate'].shape[0])

        obj_model = Model()
        model = obj_model.get_model(self.num_classes)
        iter_total = tf.Variable(0, tf.int32)
        add_iter = tf.assign_add(iter_total, 1)

        batch_size = opt.batch_size

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for ind_epoch in range(0, opt.num_epochs):
                self.logger.info('current epoch {}'.format(ind_epoch + 1))
                for ind_iter in range(0, int(num_samples_train / batch_size)):
                    uni_train, w_uni_train, targets_train = self.get_batch(data_train, num_samples_train, ind_iter * batch_size, batch_size)
                    _, cur_loss, _, cur_iter = sess.run([model['optimizer'], model['loss'], add_iter, iter_total], {
                        model['uni']: uni_train,
                        model['w_uni']: w_uni_train,
                        model['targets']: targets_train,
                        model['is_training']: True
                    })

                    if cur_iter % opt.step_display == 0:
                        self.logger.info('cur_iter {} cur_loss {:.4f}'.format(cur_iter, cur_loss))


if __name__ == '__main__':
    clsf = Classifier()
    fire.Fire({'train': clsf.train, 'predict': clsf.predict})
