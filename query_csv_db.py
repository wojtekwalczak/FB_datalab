#!/usr/bin/env python
# -*- encoding: utf-8

import sys
import numpy as np
from matplotlib import pyplot as plt

from lib.examine_csv_db import ExamineCsvDB

a = ExamineCsvDB('data/user_types/fb_user_types.csv.gz',
                 'data/user_types/feature_names.msg.gz',
                 'data/user_types/factor_names.msg.gz',
                 features_to_drop=('activated', 'activated_FB_Mobile', 'added_a_life_event', 'asked', 'claimed_an_offer', 'clipped_an_offer', 'commented_on', 'created_a_page', 'has_worked_on', 'invited', 'is_expecting', 'was_with', 'was_in', 'was_at', 'using', 'uploaded_a_file', 'unknown', 'took', 'subscribed_to', 'started_playing', 'replied_to', 'rated', 'posted_an_event', 'posted_a_video', 'posted_a_photo', 'playing', 'now_works_at', 'left', 'knows', 'just_signed_up_for', 'just_got', 'is_following'))


# remove outliers
for acol in a.feature_names:
   a.data = a.data[np.abs(a.data.ix[:, acol]-a.data.ix[:, acol].mean())<=(3*a.data.ix[:, acol].std())]

print 'Factors to choose from:', a.factor_names
#grouped = a.data.groupby('age_range')
#grouped = a.data.groupby('NoF_range')
grouped = a.data.groupby('is_male')

for acol in a.feature_names:
   print acol
   z=grouped[acol].mean()
   plt.bar(z.index, z.values)
   plt.title(acol)
   plt.show()

   print '-'*80

print grouped.size()
