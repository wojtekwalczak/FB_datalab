#!/usr/bin/env python
# -*- coding: utf-8

from lib.estimate_factor import FactorEstimation

if __name__ == '__main__':

   a = FactorEstimation('data/links/links_matrix.mtx.gz',
                        'data/links/links_colnames.msg.gz',
                        'data/links/factors.msg.gz')
   a.cv('age')
   a.predict('age', results_fn='results/links_age_prediction_results.msg')


   b = FactorEstimation('data/likes/likes_matrix.mtx.gz',
                        'data/likes/likes_colnames.msg.gz',
                        'data/links/factors.msg.gz')
   b.cv('age')
   b.predict('age', results_fn='results/likes_age_prediction_results.msg')
