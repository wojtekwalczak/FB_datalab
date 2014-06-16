#!/usr/bin/env python
# -*- encoding: utf-8

import gzip
import msgpack
import pandas as pd
from utils.utils import Utils


class GenericCsvDB(Utils):

   def init(self):
      self.data = pd.read_csv(self._matrix_fn, compression='gzip')
      self.factor_names = self._load_pickle(self._factors_fn)
      self.feature_names = self._load_pickle(self._colnames_fn)


   def reset(self):
      self.init()


   @property
   def fac_len(self):
      return len(self.factor_names)


   def drop_features(self, feats):
      for cn in feats:
         self.data = self.data.drop(cn, axis=1)
         self.feature_names.pop(self.feature_names.index(cn))
