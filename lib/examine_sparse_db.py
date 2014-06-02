#!/usr/bin/env python
# -*- encoding: utf-8

import gzip
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import brewer2mpl

from pylab import get_cmap
from collections import defaultdict

from utils.utils import Utils


class ExamineSparseDB(Utils):
   def __init__(self,
                matrix_fn=None,
                colnames_fn=None,
                factors_fn=None,
                verbose=1):

      self._matrix_fn = matrix_fn
      self._colnames_fn = colnames_fn
      self._factors_fn = factors_fn

      self.factors = None
      self.fac_len = 0

      self.verbose = verbose

      self.data = None

      self._init()

   def _init(self):
      self.data = sio.mmread(gzip.open(self._matrix_fn)).tolil()
      self.factors = self._load_pickle(self._factors_fn)
      self.fac_len = len(self.factors)
      self.col_names = self.factors + self._load_pickle(self._colnames_fn)
      assert self.data.shape[1] == len(self.col_names),\
                 'Mismatch between the number of columns: %s - %s.'\
                     % (self.data.shape[1], len(self.col_names))

   def reset(self):
      self._init()


   def _iter_features(self):
      for ind, colname in enumerate(self.col_names):
         if ind < self.fac_len: # omit columns containing factors
            continue
         yield ind, colname


   def get_popular(self, num=10, least=False):
      """
         Arguments:
          num (int) - how many features to return
          least (bool) - if True, get 'num' least popular features
      """

      m_csc = self.data.tocsc()
      f_dict = defaultdict(int)

      for ind, colname in self._iter_features():
         f_dict[colname] = m_csc.getcol(ind).nnz

      if least:
         return sorted(f_dict.items(), key=lambda x: x[1], reverse=True)[-num:]
      else:
         return sorted(f_dict.items(), key=lambda x: x[1], reverse=True)[:num]


   def get_by_freq(self, freq=100, less_than=False):
      """
         Arguments:
          freq (int) - return features which occur at least 'freq' times
          less_than (bool) - if True, return features which occur no more than
                             'freq' times
      """

      m_csc = self.data.tocsc()
      f_dict = defaultdict(int)

      for ind, colname in self._iter_features():
         occurrences = m_csc.getcol(ind).nnz

         if less_than and occurrences < freq:
            f_dict[colname] = occurrences
         elif not less_than and occurrences > freq:
            f_dict[colname] = occurrences

      return sorted(f_dict.items(), key=lambda x: x[1], reverse=True)


   def del_features_by_nam(self, names):
      new_sparse = []
      m_csc = self.data.tocsc()

      for colname in self.col_names:
         if colname not in names:
            col_index = self.col_names.index(colname)
            new_sparse.append(m_csc.getcol(col_index))

      return ([i for i in self.col_names if i not in names],
              sparse.hstack(new_sparse).tocoo())


   def del_features_by_freq(self, freq=100, less_than=False):
      names = self.factors[:]
      new_sparse = []
      m_csc = self.data.tocsc()

      for name in names:
         new_sparse.append(m_csc.getcol(self.col_names.index(name)))

      for ind, colname in self._iter_features():
         occurrences = m_csc.getcol(ind).nnz

         if less_than and occurrences > freq:
            new_sparse.append(m_csc.getcol(ind))
            names.append(colname)
         elif not less_than and occurrences < freq:
            new_sparse.append(m_csc.getcol(ind))
            names.append(colname)

      self.col_names = [i for i in self.col_names if i in names]
      self.data = sparse.hstack(new_sparse).tocoo()
      return self.data, self.col_names

   def shadow_val(self, factor, from_val, to_val):
      fac_ind = self.col_names.index(factor)

      for row_ind in range(self.data.shape[0]):
         if self.data[row_ind, fac_ind] in from_val:
            self.data[row_ind, fac_ind] = to_val

