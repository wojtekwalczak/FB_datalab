#!/usr/bin/env python
# -*- encoding: utf-8

from __future__ import division

import sys
import msgpack
import numpy as np
import scipy.io as sio

import nltk
from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

stoplist = [
   'pdc', 'ftp', 'http', 'tdnda', 'rtsp', 'vsdhngkuql',
   'file', 'ttp', 'cjuleny', 'feature', 'cache', 'wka',
   'jlc', 'lastfm', 'artists',
]

def cr_classifier(data, colnames):
   train_set = []
   clen = len(colnames)
   for cn, pos, neg in iter_neg_pos(data, colnames):
      train_set.extend([({ 'word': cn }, 'pos') for i in range(pos)])
      train_set.extend([({ 'word': cn }, 'neg') for i in range(neg)])
   return nltk.NaiveBayesClassifier.train(train_set)


def cr_tagcloud(words,
                fn,
                minsize=17,
                maxsize=50,
                size=(680, 500),
                fontname='Nobile'):

   tags = make_tags([(i[0], i[2]) for i in words],
                    minsize=minsize, maxsize=maxsize)

   create_tag_image(tags, fn, size=size, fontname=fontname)


def iter_neg_pos(data, colnames):
   clen = len(colnames)
   for ind in range(clen):
      if colnames[ind] in stoplist:
         continue
      neg, pos = np.ravel(data.getcol(ind).todense())
      yield colnames[ind], pos, neg


def get_pos_neg(data, colnames):
   res = []
   clen = len(colnames)
   for cn, _pos, _neg in iter_neg_pos(data, colnames):
      pos, neg = _pos + 0.001, _neg + 0.001
      if (neg / pos) < 0.2 or (neg / pos) > 2.0:
         res.append((cn, (neg / pos), pos if (neg / pos) < 0.2 else neg))

   res = sorted(res, key=lambda x: x[1])
   negative = sorted(res[:125], key=lambda x: x[2])
   positive = sorted(res[-125:], key=lambda x: x[2], reverse=True)

   return positive, negative


if __name__ == '__main__':
   w = open('data/emo/words_by_emo_colnames.msg')
   colnames = msgpack.unpack(w)
   w.close()

   data = sio.mmread('data/emo/words_by_emo.mtx').tocsc()

   positive, negative = get_pos_neg(data, colnames)

   cr_tagcloud(positive, 'positive_tags.png', minsize=13, maxsize=50)
   cr_tagcloud(negative, 'negative_tags.png', minsize=17, maxsize=55)

   cls = cr_classifier(data, colnames)
   cls.show_most_informative_features(10)
