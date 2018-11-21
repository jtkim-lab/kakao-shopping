#! /bin/bash
python classifier.py predict ./data/train ./model/train ./data/train/ dev kimandkang.predict.train.tsv
python evaluate.py evaluate kimandkang.predict.train.tsv ./data/train/data.h5py dev ./data/y_vocab.pkl

