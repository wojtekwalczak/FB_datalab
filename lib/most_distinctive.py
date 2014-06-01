# -*- encoding: utf-8

from __future__ import division


"""

   From among the links published by Facebook users pick up the links
   which differentiate users the most (by age, gender, location etc.).

   The input data is a sparse matrix. A number of first columns provide
   factors data (such as age, gender etc.), and the following columns
   provide features (URLs). Rows depict users.

"""

import sys
import gzip
import msgpack
import os.path
from itertools import combinations
from textwrap import wrap

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import brewer2mpl
from pylab import get_cmap


location_codes = {
   #0: u'b/d',
   1: u'500 tys.+',
   2: u'400-500 tys.',
   3: u'300-400 tys.',
   4: u'200-300 tys.',
   5: u'100-200 tys.',
   6: u'< 100 tys.',
   7: u'Wieś'
}

rs_codes = {
   0: u'Zaręczony/-a',
   1: u'To skomplikowane',
   #2: 'none',
   3: u'Wolny/-a',
   4: u'W związku',
   5: u'Małżeństwo',
   6: u'Otwarty związek',
   #7: 'in a domestic partnership',
   #8: 'divorced',
   #9: 'widowed',
   #10: 'in a civil union',
   #11: u'Separacja'
}

rs_codes = [i[1] for i in sorted(rs_codes.items(), key=lambda x: x[0])]



class ProcessMostDistinctive(object):
   """

   """

   def __init__(self,
                matrix_fn=None,
                colnames_fn=None,
                factors=None,
                verbose=1):
      """

         Args:
          matrix_fn (str): *.mtx file containing sparse data matrix

          colnames_fn (str): column names for columns in 'matrix_fn'

          factors (tuple): names of columns contaning factors (such as age).
                           The remaining columns are considered features.

          verbose (int): 0 - shut up; 1 - explain.

      """

      self.data = sio.mmread(gzip.open(matrix_fn)).tolil()

      self.factors = list(factors)
      self.fac_len = len(self.factors)

      self.col_names = self.factors + self._load_pickle(colnames_fn)

      self.verbose = verbose

      assert self.data.shape[1] == len(self.col_names),\
                 'Mismatch between the number of columns: %s - %s.'\
                     % (self.data.shape[1], len(self.col_names))

      self.df = None
      self.categorized = None


   def _reset(self):
      self.categorized = None


   def _gen_colors(self, num):
      if num < 13:
         return brewer2mpl.get_map('Set3', 'Qualitative', num).mpl_colors
      else:
         cm = get_cmap('Dark2')
         return [cm(1.*i/num) for i in range(num)]


   def _load_pickle(self, fn):
      w = gzip.open(fn)
      data = msgpack.unpack(w)
      w.close()
      return data


   def _print(self, astr):
      if self.verbose > 0:
         print astr


   def groupby(self, factor,
                     min_cum_val=10,
                     drop_factor_vals=None,
                     categorical=0,
                     drop_features=None,
                     drop_rows=None):
      """
         Group by factor 'factor'.

         Args:
          factor (str) - a factor by which the samples will be grouped.

          min_cum_val (int) - drop features for which the cumulative
                              value for largest factor's value is smaller
                              than 'min_cum_val'.

                              In other words, if for a given feature (qwe.com)
                              none of factor's categories (eg. male/female for
                              factor 'gender') is larger than 'min_cum_val',
                              then drop this feature from further analysis.

          drop_factor_vals (tuple) - drop particular values of the factor
                                     (such as 0 values for factor 'age'
                                     indicating no age for particular user).

          categorical (int) - discretize continuous values of a factor.
                              The value of 'categorical' is a number of bins.

          drop_features (tuple) - a tuple of feature names which should be
                                  dropped from analysis

          drop_rows (tuple of tuples) - drop rows by values of a particular
                                        factor. Factor 'is_full_feed' takes
                                        two values (0 and 1). To drop rows
                                        with zeros in this factor the drop_rows
                                        should be: (('is_full_feed', 0),).

         Sets:
            self.df to pandas's DataFrame object.

         Returns:
            returns pandas's DataFrame object.
      """

      assert factor in self.col_names, "Factor %s does not exist!" % (factor)

      fac_len, rows_len = self.fac_len, self.data.shape[0]
      fac_ind = self.col_names.index(factor)

      results = {}
      target_stats = {}

      m_csr = self.data.tocsr()

      if categorical:
         m_csc = self.data.tocsc()
         f_col = np.ravel(m_csc.getcol(fac_ind).todense())
         f_col_df = pd.DataFrame(f_col.tolist())
         self.categorized = pd.cut(f_col_df.ix[:, 0], categorical)


      for row_ind in range(rows_len):
         arow = np.ravel(m_csr.getrow(row_ind).todense())

         drop = False
         if drop_rows:
            for drop_factor, val_to_drop in drop_rows:
               drop_ind = self.col_names.index(drop_factor)
               if arow[drop_ind] == val_to_drop:
                  drop = True
         if drop:
            continue

         factor_val = int(arow[fac_ind])

         if drop_factor_vals and factor_val in drop_factor_vals:
            continue

         if categorical:
            factor_val = self.categorized.labels[row_ind]

         if factor_val in results:
            results[factor_val] += arow[fac_len:] # element-wise addition
            target_stats[factor_val] += 1
         else:
            results[factor_val] = arow[fac_len:]
            target_stats[factor_val] = 1

      df = pd.DataFrame(results).T

      # URLs as column names
      df.columns = self.col_names[fac_len:]

      df.insert(0, self.col_names[fac_ind], df.index.tolist())

      df.insert(1, 'users_count', [target_stats[i] for i in df.index.tolist()])

      # index from 1 to max num of rows
      df.index = range(1, df.shape[0]+1)

      if drop_features:
         df = df.drop(list(drop_features), axis=1)

      self._print(df.ix[:, :2])

      self.df = df
      return df


   def _prepare_chart_data(self):
      assert self.df is not None, "No data in 'df'. Run 'groupby()' first!"

      # count percentages for features
      df2 = self.df.ix[:, :2] # keep factors unchanged
      df2 = pd.concat([df2,
                       self.df.ix[:, 2:].div(self.df.loc[:, 'users_count'],
                                             axis=0)],
                      axis=1) # concatenate factors and features' percentages

      # count dispersion for every feature as standard deviation
      df3 = df2.ix[:, 2:].std(axis=0)
      df3.sort()

      self._print('Factor: {}'.format(df2.columns[0]))

      toplot = []
      for seq_ind in df3.index.tolist()[-9:]:
         # categories of a factor
         categories = df2.ix[:, 0]

         # percentage values for a feature indexed as seq_ind
         pct = [i*100 for i in df2.ix[:, seq_ind].values]

         # counter of rows ascribed to a given category
         ucount = [i for i in self.df.ix[:, 'users_count'].values]

         toplot.append((seq_ind, zip(categories, pct, ucount)))

         # print some stats
         self._print('   feature: {}'.format(toplot[-1][0]))
         for cat, val, u_count in toplot[-1][1]:
            self._print('     category: {}; value: {}; N={}'.format (cat,
                                                                     val,
                                                                     u_count))
      self._print('-'*80)

      return toplot

   def _autolabel(self, ax, rects, texts):
       ylim = ax.get_ylim()[1]
       for ii, rect in enumerate(rects):
           height = rect.get_height()
           ax.text(rect.get_x()+rect.get_width()/2.,
                   height + 0.05*ylim,
                   '%.2f%%'% (texts[ii]),
                   rotation='vertical',
                   color='black',
                   fontsize=7,
                   ha='center',
                   va='bottom')

   def make_chart(self,
                  xlabel=None,
                  xtick_labels=None,
                  xlabel_fontsize=7,
                  savefn=None,
                  xticks_rotation=0,
                  xticks_fontsize=6,
                  custom_colors=None,
                  bar_align='edge',
                  adjust_bottom=0.18,
                  suptitle_fontsize=13,
                  title=''):

      toplot = self._prepare_chart_data()

      fig, ax = plt.subplots(3, 3, sharey=True, sharex=True, figsize=(6, 6))
      fig.subplots_adjust(top=0.89, bottom=adjust_bottom, hspace=0.25)

      fig.suptitle(title, fontsize=suptitle_fontsize, weight='bold')

      axes = np.ravel(ax)

      categories = range(len(self.df.ix[:, 0].tolist()))
      colors = custom_colors if custom_colors\
                             else self._gen_colors(len(categories))

      if xtick_labels is None and self.categorized:
         xtick_labels = self.categorized.levels.tolist()
      elif xtick_labels is None:
         xtick_labels = [str(i) for i in categories]

      for sp_num, plot_data in enumerate(reversed(toplot)):
         values = [i[1] for i in plot_data[1]]

         axes[sp_num].set_title(plot_data[0].decode('utf-8')[:25],
                                fontsize=10)
         axes[sp_num].grid()
         rects = axes[sp_num].bar(categories,
                                  values,
                                  alpha=0.99,
                                  color=colors,
                                  align=bar_align)

         #self._autolabel(axes[sp_num], rects, values)

         if xtick_labels is not None:
            axes[sp_num].set_xticks(range(len(xtick_labels)))
            axes[sp_num].set_xticklabels(xtick_labels,
                                         fontsize=xticks_fontsize,
                                         rotation=xticks_rotation,
                                         ha='center')
         elif self.categorized:
            axes[sp_num].set_xticklabels(xtick_labels)


         for tick in axes[sp_num].yaxis.get_major_ticks():
            tick.label.set_fontsize(6)

         if sp_num in (0, 3, 6):
            axes[sp_num].set_ylabel(u'% udostępnień', fontsize=7)
         if sp_num > 5 and xlabel is not None:
            axes[sp_num].set_xlabel(xlabel, fontsize=xlabel_fontsize)


      ssizes = ' '.join(['N("%s")=%s; '\
                   % (i[0], i[1][2]) for i in zip(xtick_labels, toplot[0][1])])
      ssizes = '\n'.join(wrap(ssizes, 100))
      fig.text(0.12, 0.02, ssizes,
           backgroundcolor='white', color='black', weight='roman',
           size=6)

      if savefn is None:
         plt.show()
      else:
         plt.savefig(os.path.join('results', savefn))

      self._reset()



   def do_relationship_status(self, title, savefn, drop_features=None):
      self.groupby('relationship_status', drop_factor_vals=(2, 7, 8, 9, 10, 11))
      self.make_chart(xlabel=u'Status związku',
                   xtick_labels=rs_codes,
                   xticks_rotation=50,
                   xticks_fontsize=6,
                   adjust_bottom=0.21,
                   title=title,
                   savefn=savefn)


   def do_location_size(self, title, savefn, drop_features=None):
      self.groupby('location_size',
                drop_factor_vals=(0, ),
                drop_features=drop_features)

      self.make_chart(xlabel=u'Wielkość lokalizacji',
                   xtick_labels=location_codes.values(),
                   xticks_rotation=45,
                   suptitle_fontsize=10,
                   adjust_bottom=0.19,
                   xticks_fontsize=6,
                   title=title,
                   savefn=savefn)


   def do_hometown_size(self, title, savefn, drop_features=None):
      self.groupby('hometown_size',
                   drop_factor_vals=(0, ),
                   drop_features=drop_features)

      self.make_chart(xlabel=u'Wielkość miejsca pochodzenia',
                   xtick_labels=location_codes.values(),
                   xticks_rotation=45,
                   xlabel_fontsize=7,
                   adjust_bottom=0.19,
                   xticks_fontsize=6,
                   suptitle_fontsize=10,
                   title=title,
                   savefn=savefn)


   def do_gender(self, title, savefn, drop_features=None):
      self.groupby('is_male', drop_features=drop_features)
      self.make_chart(xlabel=u'Płeć',
                   xtick_labels=(u'Kobieta', u'Mężczyzna'),
                   custom_colors=('pink', '#6495ED'),
                   xticks_fontsize=6,
                   bar_align='center',
                   adjust_bottom=0.1,
                   title=title,
                   savefn=savefn)


   def do_age(self, title, savefn, drop_features=None):
      age_range = (0, )
      self.groupby('age_range',
                   drop_factor_vals=age_range,
                   drop_features=drop_features)

      self.make_chart(xlabel=u'Wiek',
                   xtick_labels=('13-18', '19-24', '25-30', '31-36',
                                 '37-42', '43-48', '49-54', '55-60', '61-66'),
                   xticks_rotation=45,
                   xticks_fontsize=6,
                   adjust_bottom=0.14,
                   title=title,
                   savefn=savefn)


   def do_number_of_friends(self, title, savefn):
      self.groupby('number_of_friends_range',
                drop_factor_vals=(0, 1),
                drop_rows=( ('is_full_feed', 0), ))

      self.make_chart(xlabel=u'Liczba znajomych',
                   xtick_labels=('51-100', '101-150', '151-200',
                                 '201-250', '251-300', '301-350', '351-400',
                                 '401-450', '451-500', '501+'),
                   xticks_rotation=45,
                   xticks_fontsize=6,
                   adjust_bottom=0.15,
                   title=title,
                   savefn=savefn)
