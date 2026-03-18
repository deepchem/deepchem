#! /bin/bash
python mol2vec.py > data.txt
python -m gensim.scripts.word2vec_standalone -train data.txt -output vec.txt -size 200 -sample 1e-4 -binary 0 -iter 3

