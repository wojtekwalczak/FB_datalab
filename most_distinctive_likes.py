#!/usr/bin/env python
# -*- encoding: utf-8

from lib.most_distinctive import ProcessMostDistinctive

if __name__ == '__main__':
   a = ProcessMostDistinctive(matrix_fn='data/likes/likes_matrix.mtx.gz',
                              colnames_fn='data/likes/likes_colnames.msg.gz',
                              factors_fn='data/likes/factors.msg.gz')

   a.do_age(u'Popularność lajków ze względu na wiek',
            savefn='likes_most_distinctive_age.png')

   a.do_gender(u'Popularność lajków ze względu na płeć',
               savefn='likes_most_distinctive_gender.png')

   a.do_location_size(u'Popularność lajków ze względu na wielkość'\
                       ' obecnej lokalizacji',
                      savefn='likes_most_distinctive_location.png')

   a.do_hometown_size(u'Popularność lajków ze względu na wielkość'\
                       ' miejsca pochodzenia',
                      savefn='likes_most_distinctive_hometown.png')

   a.do_number_of_friends(u'Popularność lajków ze względu na liczbę znajomych',
                          savefn='likes_most_distinctive_NoF.png')

   a.do_relationship_status(u'Popularność lajków ze względu na status związku',
                            savefn='likes_most_distinctive_relationship_status.png')

