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



