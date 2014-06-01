# -*- coding: utf-8 -*-

"""
   nai.infer_methods.py
   ~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from utils import mean_stdev, vals_by_key, vals_by_key_filter, add_node_attrs


default_settings = {
   # which node attribute to shadow
   'attr_to_del': 'age',

   # where to store the shadowed value
   'attr_backup': 'real_age',

   # how many iterations of shadow/predict actions to perform
   'max_iters': 10,

   # the value a real value will be shadowed with
   'shadow_val': -1,

   # fraction of values to shadow in a graph
   'shadow_share': 0.025,

   # minimum number of nodes to base the inference on
   'min_nodes': 2,
   'max_stdev': 5.0,
}

# iterate graph by neighbors, infer attribute by longest sequence
class NeighborsLongestSequence(object):

   settings = default_settings

   def __init__(self):
      pass


   def _longest_consecutive(self, s, max_gap=2):
      """
         Find longest sequence of values in a list. There may be gaps
         in the sequence, but not longer than 'max_gap'.

         Example:

         >>> _longest_consecutive([1, 2, 3, 4, 7, 10, 14, 14, 14, 16, 17, 19])
         [14, 14, 14, 16, 17, 19]
      """
      if not s:
         return []
      s = sorted(s)
      lists = [[s.pop(0)]]
      for elem in s:
         if lists[-1][-1] in [elem-i for i in range(0, max_gap+1)]:
            lists[-1].append(elem)
         else:
            lists.append([elem])
      return sorted(lists, key=len)[-1]

   def infer_attr(self, subgraph_data, **kwargs):
      """
      """
      sv, atd = kwargs['shadow_val'], kwargs['attr_to_del']
      min_nodes = kwargs.get('min_nodes')

      neighbors = subgraph_data['neighbors']
      root_node = subgraph_data['root_node']

      a_vals = vals_by_key_filter(neighbors, atd, sv)
      lc = self._longest_consecutive(a_vals)
      lc_len = len(lc)

      # not enough nodes to base estimation on
      if lc_len < min_nodes or lc_len == 0:
         return [ (None, None, None) ]

      # 'min_nodes' is set to 1, and lc_len==1, so simply return
      # the result for a single node
      if lc_len == 1:
         return [ (root_node, lc[0], 1) ]

      val_predicted, stdev, prediction_base = mean_stdev(lc)

      if prediction_base >= min_nodes and stdev < kwargs.get('max_stdev'):
         return [ (root_node, val_predicted, prediction_base) ]

      return [ (None, None, None) ]


   def graph_iter(self, G, **kwargs):
      for root_node in G.nodes_iter():
         if G.node[root_node][kwargs['attr_to_del']] != kwargs['shadow_val']:
            continue
         neighbors = add_node_attrs(G.neighbors(root_node), G)
         yield { 'neighbors': neighbors, # a list of neighbors
                 'root_node': root_node }



# iterate by cliques containing node, infer by largest and most homogenous clique
class MostHomogenousClique(object):

   settings = default_settings

   def __init__(self):
      pass

   def graph_iter(self, G, **kwargs):
      sv = kwargs['shadow_val']
      for root_node in G.nodes_iter():
         n_freq = defaultdict(int)

         if G.node[root_node][kwargs['attr_to_del']] != sv:
            continue

         cliques = []

         for aclique_raw in nx.cliques_containing_node(G, nodes=root_node):
            cliques.append(add_node_attrs(aclique_raw, G))

         yield { 'subgraph': cliques,
                 'root_node': root_node }


   def infer_attr(self, subgraph_data, **kwargs):
      """
         Counts mean and standard deviation for 'attr_to_del' (eg. age)
         and treats the mean as predicted value if standard
         deviation < kwargs['stdev'].
      """

      sv, atd = kwargs['shadow_val'], kwargs['attr_to_del']
      min_nodes, max_stdev = kwargs.get('min_nodes'), kwargs.get('max_stdev')

      subgraphs = subgraph_data['subgraph']
      root_node = subgraph_data['root_node']

      sg = []
      for subgraph in subgraphs:
         a_vals = vals_by_key_filter(subgraph, atd, sv)
         vals_len = len(a_vals)

         # not enough nodes to base estimation on
         if vals_len < min_nodes or vals_len == 0:
            return [ (None, None, None) ]

         if vals_len == 1: # only a single node in a subgraph
            sg.append((a_vals[0], 100, 1))
            continue

         val_predicted, val_stdev, prediction_base = mean_stdev(a_vals, null_val=sv)
         if prediction_base >= min_nodes and val_stdev < max_stdev:
            sg.append((val_predicted, val_stdev, prediction_base))

      # sort by homogenity, and then by size
      sg = sorted(sg, key=lambda x: (x[1], x[2]))

      if not sg:
         return [ (None, None, None) ]

      return [(root_node, sg[0][0], sg[0][2])]



################################################################################

available_methods = {
   'NeighborsLongestSequence': NeighborsLongestSequence,
   'MostHomogenousClique': MostHomogenousClique,
#   'default': MostHomogenousClique,
   'default': NeighborsLongestSequence,
}
