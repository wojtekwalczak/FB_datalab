#!/usr/bin/env python
# -*- encoding: utf-8

import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import brewer2mpl

from pylab import get_cmap
from collections import defaultdict

from lib.generic_csv_db import GenericCsvDB

pd.set_option('display.max_columns', None)

class ExamineCsvDB(GenericCsvDB):
   def __init__(self,
                matrix_fn=None,
                colnames_fn=None,
                factors_fn=None,
                features_to_drop=None,
                verbose=1):

      self._matrix_fn = matrix_fn
      self._colnames_fn = colnames_fn
      self._factors_fn = factors_fn

      self.factors = None

      self.verbose = verbose

      self.data = None

      self.init()

      if features_to_drop:
         self.drop_features(features_to_drop)

      # absolute values to fractions of all of the activities
      self.data['feed_total'] = self.data.ix[:, self.feature_names].sum(axis=1)
      df = pd.concat([self.data.ix[:, self.factor_names],
                       self.data.ix[:, self.feature_names]\
                                      .div(self.data.ix[:, 'feed_total'], axis=0),
                       pd.DataFrame({'feed_total': self.data.ix[:, 'feed_total']})],
                                                     axis=1)

      self.data = df

