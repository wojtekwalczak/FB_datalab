#!/usr/bin/env python
# -*- encoding: utf-8

import gzip
import scipy.io as sio
from utils.utils import Utils


class GenericSparseDB(Utils):

   def init(self):
      self.data = sio.mmread(gzip.open(self._matrix_fn)).tolil()
      self.factors = self._load_pickle(self._factors_fn)
      self.fac_len = len(self.factors)
      self.col_names = self.factors + self._load_pickle(self._colnames_fn)
      assert self.data.shape[1] == len(self.col_names),\
                 'Mismatch between the number of columns: %s - %s.'\
                     % (self.data.shape[1], len(self.col_names))

   def reset(self):
      self.init()
