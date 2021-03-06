FB_datalab
==========

**Python scripts + Facebook data**


most_distinctive_links.py
-------------------------

From among the links published by Facebook users pick up the links
which differentiate users the most (by age, gender, location etc.).

The input data is a sparse matrix. A number of first columns provide
factors data (such as age, gender etc.), and the following columns
provide features (URLs). Rows depict users.

Produce charts like the one below:

![alt tag](http://laboratoriumdanych.pl/wp-content/uploads/2014/05/most_distinctive_age.png)

most_distinctive_likes.py
-------------------------

Similar to most_distinctive_links.py, but works on Facebook Pages
liked by particular users.

run_nai.py
----------

Cross-validate various methods of node attribute inference

Simplest use case:

```python
from lib.nai.cv import NodeAttrInferCV
z = NodeAttrInferCV('data/fb_friends_graph.gml.gz')
z.run()
z.simple_stats()
```

`simple_stats()` method will return an output similar to this:

```
Shadowed per iteration: 251
Unshadowed: [251, 251, 248]
--------------------------------------------------------------------------------
iteration     1     2     3
diff                       
0          0.28  0.33  0.24
1          0.20  0.17  0.22
2          0.13  0.14  0.14
3          0.07  0.07  0.09
4          0.06  0.03  0.06

[5 rows x 3 columns]
--------------------------------------------------------------------------------
iteration     1     2     3
diff                       
0          0.28  0.33  0.24
1          0.48  0.50  0.46
2          0.61  0.64  0.60
3          0.68  0.71  0.69
4          0.74  0.74  0.75

[5 rows x 3 columns]
--------------------------------------------------------------------------------
Iteration: 1. Top 10 biggest errors: [41, 38, 29, 29, 27, 25, 25, 25, 24, 20].
Iteration: 2. Top 10 biggest errors: [46, 39, 36, 35, 32, 30, 29, 28, 28, 27].
Iteration: 3. Top 10 biggest errors: [35, 34, 32, 29, 25, 21, 19, 18, 17, 16].
```


query_sparse_db.py
------------------

This package comes with a number of databases in scipy's sparse matrix format. query_sparse_db.py provides some examples of helper functions to facilitate examining of these databases.

For example you are able to do this:

```
from lib.examine_sparse_db import ExamineSparseDB

a = ExamineSparseDB('data/links/links_matrix.mtx.gz',
                    'data/links/links_colnames.msg.gz',
                    'data/links/factors.msg.gz')

print a.get_popular(10)
```

This will return 10 most popular features:

```
[('youtube.com', 14980), ('Obrazki FB', 13843), ('facebook.com', 10432), ('demotywatory.pl', 7316), ('kwejk.pl', 6677), ('gazeta.pl', 5513), ('vimeo.com', 5321), ('onet.pl', 5031), ('wrzuta.pl', 4482), ('wp.pl', 3925)]
```

On the other hand calling:

```
print a.get_popular(10, least=True)
```

will return 10 least popular links:

```
[('548426_440524952689441_220491277_n.jpg', 16), ('523004_292705267495944_1723994932_n.jpg', 16), ('402728_357842164303281_1435223383_n.jpg', 15), ('christian-dogma.com', 15), ('youtube.com/watch?v=BVp8xWsteMo', 15), ('start.no', 15), ('lostbubblegame.com', 15), ('pato.pl', 15), ('takafaza.pl', 14), ('jakboczek.pl', 13)]
```


run_estimations.py
------------------

A sample of how to carry out cross-validation and prediction of particular factors given the available set of features.

To perform cross-valdation for predicting 'age' based on links Facebook users post on their profiles one should do the following:

```
from lib.estimate_factor import FactorEstimation

a = FactorEstimation('data/links/links_matrix.mtx.gz',
                     'data/links/links_colnames.msg.gz',
                     'data/links/factors.msg.gz')

a.cv('age', # factor to predict
     'split_val': 0, # values of factor to predict
     'shadow_func': lambda x: x > 66, # for users older than 66...
     'shadow_to_val': 0, # shadow their age with 0 and predict age
     'del_freq': 450) # remove features if shared by 450 users
```

This will produce results similar to the following lines:

```
4.14232033819 +/- 5.07028463257
4.3597933302 +/- 5.63557438023
3.99154532644 +/- 5.25908584676
4.04088345865 +/- 5.13718948072
4.0469924812 +/- 5.30592752445
```

These are mean errors and standard deviations for predictions in five iterations.


run_emotions_detection.py
-------------------------

Generate tagclouds of words with positive/negative meaning (words mostly in Polish), such as:

![alt tag](http://laboratoriumdanych.pl/wp-content/uploads/2014/06/negative_tags.png)

Also, generate a list of most distinctive features using NLTK's `nltk.NaiveBayesClassifier`.
