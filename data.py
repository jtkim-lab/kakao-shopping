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
os.environ['OMP_NUM_THREADS'] = '1'
import re
import sys
import pickle
import traceback
from collections import Counter
from multiprocessing import Pool

import tqdm
import fire
import h5py
import numpy as np
import mmh3
from keras.utils.np_utils import to_categorical

from misc import get_logger, Option
opt = Option('./config.json')

re_sc = re.compile(r'[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')


class Reader(object):
    def __init__(self, data_path_list, div, offset_begin, end_offset):
        self.data_path_list = data_path_list
        self.div = div
        self.offset_begin = offset_begin
        self.end_offset = end_offset

    def is_range(self, i):
        if self.offset_begin is not None and i < self.offset_begin:
            return False
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

    def get_size(self):
        offset = 0
        count = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[self.div]['pid'].shape[0]
            if not self.offset_begin and not self.end_offset:
                offset += sz
                count += sz
                continue
            if self.offset_begin and offset + sz < self.offset_begin:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                count += 1
            offset += sz
        return count

    def get_class(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    def generate(self):
        offset = 0
        for data_path in self.data_path_list:
            h = h5py.File(data_path, 'r')[self.div]
            sz = h['pid'].shape[0]
            if self.offset_begin and offset + sz < self.offset_begin:
                offset += sz
                continue
            if self.end_offset and self.end_offset < offset:
                break
            for i in range(sz):
                if not self.is_range(offset + i):
                    continue
                class_name = self.get_class(h, i)
                yield h['pid'][i], class_name, h, i
            offset += sz

    def get_y_vocab(self, data_path):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            class_name = self.get_class(h, i)
            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def preprocessing(data):
    try:
        cls, data_path_list, div, out_path, begin_offset, end_offset = data
        data = cls()
        data.load_y_vocab()
        data.preprocessing(data_path_list, div, begin_offset, end_offset, out_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def build_y_vocab(data):
    try:
        path_data, div = data
        reader = Reader([], div, None, None)
        y_vocab = reader.get_y_vocab(path_data)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


class Data:
    path_y_vocab = os.path.join(opt.path_data, 'y_vocab.pkl')
    tmp_chunk_tpl = 'tmp/base.chunk.%s'

    def __init__(self):
        self.logger = get_logger('data')

    def load_y_vocab(self):
        self.y_vocab = pickle.loads(open(self.path_y_vocab, 'rb').read())

    def build_y_vocab(self):
        if not os.path.exists(opt.path_data):
            os.makedirs(opt.path_data)

        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab,
                [(data_path, 'train') for data_path in opt.train_data_list]).get(9999999)
            pool.close()
            pool.join()
            y_vocab = set()
            for _y_vocab in rets:
                for k in _y_vocab:
                    y_vocab.add(k)
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        self.logger.info('size of y_vocab {}'.format(len(self.y_vocab)))
        pickle.dump(self.y_vocab, open(self.path_y_vocab, 'wb'))

    def _split_data(self, data_path_list, div, chunk_size):
        total = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
        chunks = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]
        return chunks

    def preprocessing(self, data_path_list, div, begin_offset, end_offset, out_path):
        self.div = div
        reader = Reader(data_path_list, div, begin_offset, end_offset)
        rets = []
        for pid, label, h, i in reader.generate():
            y, x = self.parse_data(label, h, i)
            if y is None:
                continue
            rets.append((pid, y, x))
        self.logger.info('sz {}'.format(len(rets)))
        pickle.dump(rets, open(out_path, 'wb'))
        self.logger.info('{}-{} (size {})'.format(begin_offset, end_offset, end_offset - begin_offset))

    def _preprocessing(self, cls, data_path_list, div, chunk_size):
        chunk_offsets = self._split_data(data_path_list, div, chunk_size)
        num_chunks = len(chunk_offsets)
        self.logger.info('split data into {} chunks, # of classes {}'.format(num_chunks, len(self.y_vocab)))
        pool = Pool(opt.num_workers)
        try:
            pool.map_async(preprocessing, [(cls, data_path_list, div, self.tmp_chunk_tpl % cidx, begin, end) for cidx, (begin, end) in enumerate(chunk_offsets)]).get(9999999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        return num_chunks

    def get_words(self, str_target, ind):
        str_all = str_target[ind]
        str_all = str_all.decode('utf-8')
        str_all = re_sc.sub(' ', str_all).strip().split()
        words = [elem_word.strip() for elem_word in str_all]
        words = [elem_word for elem_word in words if len(elem_word) >= opt.min_word_length and len(elem_word) < opt.max_word_length]
        if not words:
            return [None] * 2

        hash_func = lambda x: mmh3.hash(x, seed=42)
        x = [hash_func(elem_word) % opt.unigram_hash_size + 1 for elem_word in words]
        return x

    def get_price(self, price, ind):
        # TODO (Jungtaek): if we would like to use price feature, we need to find smarter way to normalize it.
        return price[ind] / 10000.0

    def parse_data(self, label, h, i):
        # h: ['bcateid', 'brand', 'dcateid', 'img_feat', 'maker', 'mcateid', 'model', 'pid', 'price', 'product', 'scateid', 'updttm']
        Y = self.y_vocab.get(label)
        if Y is None and self.div in ['dev', 'test']:
            Y = 0
        if Y is None and self.div != 'test':
            return [None] * 2
        Y = to_categorical(Y, len(self.y_vocab))

        x = []
        # TODO (Jungtaek): make sure whether or not they are appropriate features
        list_texts = [h['product'], h['maker'], h['brand'], h['model']]
        for elem in list_texts:
            x += self.get_words(elem, i)
        xv = Counter(x).most_common(opt.max_len)

        x = np.zeros(opt.max_len, dtype=np.float32)
        v = np.zeros(opt.max_len, dtype=np.int32)
        for i in range(len(xv)):
            x[i] = xv[i][0]
            v[i] = xv[i][1]

        price = self.get_price(h['price'], i)
        print(price)
        img_feat = np.zeros(opt.len_img_feat)
#        img_feat = np.array(h['img_feat'][:opt.len_img_feat])
        return Y, (x, v, price, img_feat)

    def create_dataset(self, g, size, num_classes):
        shape_w = (size, opt.max_len)
        shape_img = (size, opt.len_img_feat)

        g.create_dataset('uni', shape_w, chunks=True, dtype=np.int32)
        g.create_dataset('w_uni', shape_w, chunks=True, dtype=np.float32)
        g.create_dataset('img_feat', shape_img, chunks=True, dtype=np.float32)
        g.create_dataset('price', (size,), chunks=True, dtype=np.float32)
        g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('pid', (size,), chunks=True, dtype='S12')

    def init_chunk(self, chunk_size, num_classes):
        chunk_shape_w = (chunk_size, opt.max_len)
        chunk_shape_img = (chunk_size, opt.len_img_feat)

        chunk = {}
        chunk['uni'] = np.zeros(shape=chunk_shape_w, dtype=np.int32)
        chunk['w_uni'] = np.zeros(shape=chunk_shape_w, dtype=np.float32)
        chunk['img_feat'] = np.zeros(shape=chunk_shape_img, dtype=np.float32)
        chunk['price'] = []
        chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
        chunk['pid'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset, with_pid_field=False):
        num = chunk['num']
        dataset['uni'][offset:offset + num, :] = chunk['uni'][:num]
        dataset['w_uni'][offset:offset + num, :] = chunk['w_uni'][:num]
        dataset['img_feat'][offset:offset + num, :] = chunk['img_feat'][:num]
        dataset['price'][offset:offset + num] = chunk['price'][:num]
        dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        if with_pid_field:
            dataset['pid'][offset:offset + num] = chunk['pid'][:num]

    def copy_bulk(self, A, B, offset, y_offset, with_pid_field=False):
        num = B['cate'].shape[0]
        y_num = B['cate'].shape[1]
        A['uni'][offset:offset + num, :] = B['uni'][:num]
        A['w_uni'][offset:offset + num, :] = B['w_uni'][:num]
        A['img_feat'][offset:offset + num, :] = B['img_feat'][:num]
        A['price'][offset:offset + num] = B['w_uni'][:num]
        A['cate'][offset:offset + num, y_offset:y_offset + y_num] = B['cate'][:num]
        if with_pid_field:
            A['pid'][offset:offset + num] = B['pid'][:num]

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def make_db(self, data_name, output_dir='data/train', train_ratio=0.8):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(opt.path_tmp):
            os.makedirs(opt.path_tmp)

        if data_name == 'train':
            str_div = 'train'
            data_path_list = opt.train_data_list 
        elif data_name == 'dev':
            str_div = 'dev'
            data_path_list = opt.dev_data_list 
        elif data_name == 'test':
            str_div = 'test'
            data_path_list = opt.test_data_list
        else:
            assert False, '{} is not valid.'.format(data_name)

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        np.random.seed(42)
        self.logger.info('make database from {} dataset with train_ratio {}'.format(data_name, train_ratio))

        self.load_y_vocab()
        num_input_chunks = self._preprocessing(
            Data,
            data_path_list,
            str_div,
            chunk_size=opt.chunk_size
        )

        fout_data = h5py.File(os.path.join(output_dir, 'data.h5py'), 'w')

        reader = Reader(data_path_list, str_div, None, None)
        tmp_size = reader.get_size()
        train_indices, train_size = self.get_train_indices(tmp_size, train_ratio)

        dev_size = tmp_size - train_size
        if all_dev:
            train_size = 1
            dev_size = tmp_size
        if all_train:
            train_size = tmp_size
            dev_size = 1

        train = fout_data.create_group('train')
        dev = fout_data.create_group('dev')
        self.create_dataset(train, train_size, len(self.y_vocab))
        self.create_dataset(dev, dev_size, len(self.y_vocab))
        self.logger.info('train_size {} dev_size {}'.format(train_size, dev_size))

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk_size = opt.db_chunk_size
        chunk = {
            'train': self.init_chunk(chunk_size, len(self.y_vocab)),
            'dev': self.init_chunk(chunk_size, len(self.y_vocab))
        }
        chunk_order = list(range(num_input_chunks))
        np.random.shuffle(chunk_order)
        for input_chunk_idx in chunk_order:
            path = os.path.join(self.tmp_chunk_tpl % input_chunk_idx)
            self.logger.info('process {}'.format(path))
            data = list(enumerate(pickle.loads(open(path, 'rb').read())))
            np.random.shuffle(data)
            for data_idx, (pid, y, vw) in data:
                if y is None:
                    continue
                v, w, price, img_feat = vw
                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True
                if v is None:
                    continue
                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['uni'][idx] = v
                c['w_uni'][idx] = w
                c['price'].append(price)
                c['img_feat'][idx] = img_feat
                c['cate'][idx] = y
                c['num'] += 1
                if not is_train:
                    # TODO (Jungtaek): why it needs to convert to bytes object. I am not sure, but it is meaningless.
                    c['pid'].append(np.string_(pid))
                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t], with_pid_field=(t == 'dev'))
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size, len(self.y_vocab))
            sample_idx += len(data)
        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t], with_pid_field=(t == 'dev'))
                num_samples[t] += chunk[t]['num']

        for cur_div in ['train', 'dev']:
            ds = dataset[cur_div]
            size = num_samples[cur_div]
            shape_w = (size, opt.max_len)
            shape_img = (size, opt.len_img_feat)
            ds['uni'].resize(shape_w)
            ds['w_uni'].resize(shape_w)
            ds['img_feat'].resize(shape_img)
            ds['price'].resize((size,))
            ds['cate'].resize((size, len(self.y_vocab)))

        fout_data.close()
        meta = {'y_vocab': self.y_vocab}
        pickle.dump(meta, open(os.path.join(output_dir, 'meta'), 'wb'))

        self.logger.info('# of classes %s' % len(meta['y_vocab']))
        self.logger.info('# of samples in train %s' % num_samples['train'])
        self.logger.info('# of samples in dev %s' % num_samples['dev'])
        self.logger.info('data %s' % os.path.join(output_dir, 'data.h5py'))
        self.logger.info('meta %s' % os.path.join(output_dir, 'meta'))

if __name__ == '__main__':
    data = Data()
    fire.Fire({
        'make_db': data.make_db,
        'build_y_vocab': data.build_y_vocab
    })
