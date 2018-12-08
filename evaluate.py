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
import pickle

import fire
import h5py
import numpy as np


def evaluate(predict_path, data_path, str_div, path_y_vocab):
    h = h5py.File(data_path, 'r')[str_div]
    y_vocab = pickle.loads(open(path_y_vocab, 'rb').read())
    inv_y_vocab = {
        v: k for k, v in y_vocab.items()
    }
    fin = open(predict_path, 'rb')
    hit = {'b': 0, 'm': 0, 's': 0, 'd': 0}
    n = {'b': 0, 'm': 0, 's': 0, 'd': 0}

    print('[INFO] load ground-truth...')
    CATE = np.argmax(h['cate'], axis=1)
    for p, y in zip(fin, CATE):
        p = p.decode('utf-8')
        pid, b, m, s, d = p.split('\t')
        b, m, s, d = list(map(int, [b, m, s, d]))
        gt = list(map(int, inv_y_vocab[y].split('>')))
        for depth, _p, _g in zip(['b', 'm', 's', 'd'], [b, m, s, d], gt):
            if _g == -1:
                continue
            n[depth] = n.get(depth, 0) + 1
            if _p == _g:
                hit[depth] = hit.get(depth, 0) + 1
    for d in ['b', 'm', 's', 'd']:
        if n[d] > 0:
            print('[INFO] {} accuracy: {:.3f}({}/{})'.format(d, hit[d] / float(n[d]), hit[d], n[d]))
    score = sum([hit[d] / float(n[d]) * w for d, w in zip(['b', 'm', 's', 'd'],[1.0, 1.2, 1.3, 1.4])]) / 4.0
    print('[INFO] score: %.3f' % score)

if __name__ == '__main__':
    fire.Fire({'evaluate': evaluate})
