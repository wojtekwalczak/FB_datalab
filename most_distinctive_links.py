# -*- encoding: utf-8

from lib.most_distinctive import ProcessMostDistinctive

if __name__ == '__main__':
   a = ProcessMostDistinctive(matrix_fn='data/links/links_matrix.mtx.gz',
                              colnames_fn='data/links/links_colnames.msg.gz',
                              factors_fn='data/links/factors.msg.gz')

   a.do_age(u'Popularność linków ze względu na wiek',
            savefn='links_most_distinctive_age.png')

   a.do_gender(u'Popularność linków ze względu na płeć',
               savefn='links_most_distinctive_gender.png')

   a.do_location_size(u'Popularność linków ze względu na wielkość'\
                       ' obecnej lokalizacji',
                      savefn='links_most_distinctive_location.png')

   a.do_hometown_size(u'Popularność linków ze względu na wielkość'\
                       ' miejsca pochodzenia',
                      savefn='links_most_distinctive_hometown.png')

   a.do_number_of_friends(u'Popularność linków ze względu na liczbę znajomych',
                          savefn='links_most_distinctive_NoF.png')

   a.do_relationship_status(u'Popularność linków ze względu na status związku',
                            savefn='links_most_distinctive_relationship_status.png')

