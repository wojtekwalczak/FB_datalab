#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import division

import sys
import gzip
import msgpack

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import brewer2mpl
from pylab import get_cmap

from scipy.sparse import linalg
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier as KNNC

from lib.examine_sparse_db import ExamineSparseDB


class FactorEstimation(ExamineSparseDB):
   def __init__(self, matrix_fn, colnames_fn, factors_fn, verbose=1):
      self._matrix_fn = matrix_fn
      self._colnames_fn = colnames_fn
      self._factors_fn = factors_fn

      self.data = None
      self.factors = None
      self.fac_len = 0
      self.col_names = None

      self.verbose = verbose

      self.clf = None
      self.null_set, self.non_null_set = None, None

      self.init()


   def _split_data(self, factor, factor_null_val):
      """
         Splits self.data into two sparse matrices.

         Arguments:

         Returns:
          A tuple of two sparse matrices containing rows for which
          factor's 'factor' values:
          (1) equal 'factor_null_val'
          (2) do not equal 'factor_null_val'

      """
      fac_len, rows_len = self.fac_len, self.data.shape[0]
      fac_ind = self.col_names.index(factor)

      non_null_set = []
      null_set = []

      m_csr = self.data.tocsr()
      for row_ind in range(rows_len):
         arow = np.ravel(m_csr.getrow(row_ind).todense())
         if arow[fac_ind] == factor_null_val:
            null_set.append(m_csr.getrow(row_ind))
         else:
            non_null_set.append(m_csr.getrow(row_ind))

      return (sparse.vstack(null_set).tolil(),
              sparse.vstack(non_null_set).tolil())


   def _iter_features(self):
      """
         Returns a generator, which iterates through the values of fetures.

         Yields a tuple of (column_index, column_name).
      """
      for ind, colname in enumerate(self.col_names):
         if ind < self.fac_len: # omit columns containing factors
            continue
         yield ind, colname


   def _get_features_only(self, from_m=None):
      """
         Drop factors data from the matrix and return features-only data.

         Arguments:
          from_m (sparse matrix) - if set, drop factors from 'from_m'.
                                   Otherwise, drop from 'self.data'.
      """
      new = []
      m_csc = None
      if from_m is None:
         m_csc = self.data.tocsc()
      else:
         m_csc = from_m.tocsc()

      for ind, colname in self._iter_features():
         new.append(m_csc.getcol(ind).tocoo())
      return sparse.hstack(new)



   def _prepare(self, factor,
                      split_val,
                      shadow_func=None,
                      shadow_to_val=None,
                      del_freq=None):

      if shadow_func is not None:
         self.shadow_val(factor, shadow_func, shadow_to_val)

      if del_freq is not None:
         self.data, self.col_names = self.del_features_by_freq(freq=del_freq)
         #self.data, self.col_names = self.del_features_by_freq(freq=30, less_than=True)
      self.null_set, self.non_null_set = self._split_data(factor, split_val)



   def cv(self, factor,
                split_val,
                shadow_func=None,
                shadow_to_val=None,
                del_freq=None):
      """
         Cross-validate prediction of factor 'factor'.
      """

      self._prepare(factor,
                    split_val,
                    shadow_func=shadow_func,
                    shadow_to_val=shadow_to_val,
                    del_freq=del_freq)

      fac_ind = self.col_names.index(factor)
      self.clf = KNNC(40, algorithm='brute', metric='cosine')
      z=self._get_features_only(self.non_null_set).astype(float)
      target = np.ravel(self.non_null_set.getcol(fac_ind).todense())
      u, s, v = linalg.svds(z, k=51)
      T = u.dot(np.diag(s))

      kf = cross_validation.KFold(len(target), 5)
      for train_idx, test_idx in kf:
         #print len(train_idx), len(test_idx)
         self.clf.fit(T[train_idx], target[train_idx])
         r = self.clf.predict(T[test_idx])
         print np.mean(np.abs(r - target[test_idx])),\
               "+/-",\
               np.std(np.abs(r - target[test_idx]))
         #for predicted, real in zip(r, target[test_idx]):
         #   if real > 45.0:
         #      print round(predicted, 0), round(real, 0)



   def predict(self, factor,
                     split_val,
                     shadow_func=None,
                     shadow_to_val=None,
                     del_freq=None,
                     results_fn=None):

      self._prepare(factor,
                    split_val,
                    shadow_func=shadow_func,
                    shadow_to_val=shadow_to_val,
                    del_freq=del_freq)

      fac_ind = self.col_names.index(factor)
      self.clf = KNNC(40, algorithm='brute', metric='cosine')
      z=self._get_features_only(self.non_null_set).astype(float)
      target = np.ravel(self.non_null_set.getcol(fac_ind).todense())
      u, s, v = linalg.svds(z, k=51)
      T = u.dot(np.diag(s))

      z2=self._get_features_only(self.non_null_set).astype(float)
      u2, s2, v2 = linalg.svds(z2, k=51)
      T2 = u2.dot(np.diag(s2))


      results = []
      self.clf.fit(T, target)
      for row_ind in range(self.null_set.shape[0]):
         r = self.clf.predict(T2[row_ind])
         results.append((int(self.non_null_set[row_ind, 0]), int(r[0])))

      if results_fn is not None:
         w = open(results_fn, 'w')
         msgpack.pack(results, w)
         w.close()
      else:
         print results

