#!/usr/bin/env python
# -*- coding: utf-8

from lib.estimate_factor import FactorEstimation

if __name__ == '__main__':

   kwargs = {
      'split_val': 0,
      'shadow_func': lambda x: x > 66,
      'shadow_to_val': 0,
      'del_freq': 450
   }

   a = FactorEstimation('data/links/links_matrix.mtx.gz',
                        'data/links/links_colnames.msg.gz',
                        'data/links/factors.msg.gz')
   a.cv('age', **kwargs)

   a.predict('age',
             results_fn='results/links_age_prediction_results.msg',
             **kwargs)


   b = FactorEstimation('data/likes/likes_matrix.mtx.gz',
                        'data/likes/likes_colnames.msg.gz',
                        'data/links/factors.msg.gz')
   b.cv('age', **kwargs)

   b.predict('age',
             results_fn='results/likes_age_prediction_results.msg',
             **kwargs)
