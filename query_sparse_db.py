#!/usr/bin/env python
# -*- encoding: utf-8


from lib.examine_sparse_db import ExamineSparseDB

a = ExamineSparseDB('data/links/links_matrix.mtx.gz',
                    'data/links/links_colnames.msg.gz',
                    'data/links/factors.msg.gz')

print a.get_popular(50)
#print a.get_popular(50, least=True)
#print a.get_by_freq()

#col_names, data = a.del_features_by_nam(['Kwejk', 'Obrazki FB', 'youtube.com', 'facebook.com'])
#print len(col_names), data.shape

col_names, data = a.del_features_by_freq()
print len(col_names), data.shape

