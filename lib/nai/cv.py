# -*- coding: utf-8 -*-

"""
   nai/cv.py
   ~~~~~~~~~

   Cross-validate the accuracy of node attribute inference in a graph.

"""

import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from csv import QUOTE_ALL
from random import sample
from collections import namedtuple, defaultdict

from utils import add_node_attrs, vals_by_key, mean_stdev, read_gml
from infer_methods import available_methods

class HandleMethods(object):

   def add_method(self, method, method_data):
      """
         Add custom method 'method' to the set of available methods.
      """
      self._methods[method] = method_data


   def list_methods(self):
      """
         List available methods
      """
      return [i for i in self._methods.keys() if i != 'default']


   def get_current_method(self):
      return self._methods[self._method]


   def register_method(self, method):
      """
         Register a method 'method'.
      """

      if not self._methods.get(method, False):
         raise KeyError('Method {} does not exist!'.format(method))

      self._method = method
      self._method_instance = self._methods[method]()
      self._register_settings(method)



class HandleSettings(object):
   """

      Handle settings for particular method.

   """

   def _register_settings(self, method):
      self._settings = self._methods[method].settings


   def get_setting(self, setting):
      return self._settings.get(setting, None)


   def set_setting(self, key, val):
      self._settings[key] = val



class CVGraphPreparation(object):
   """

      Prepare a graph for cross validation, ie. remove random
      attributes.

   """

   def _del_random_attributes(self):
      self._test_G = self._G.copy()

      shadow_share = self.get_setting('shadow_share')
      shadow_val = self.get_setting('shadow_val')

      Gn = self._G.nodes()

      for nn in sample(Gn, int(shadow_share*len(Gn))):

         # don't shadow the attribute twice for the same node,
         # since it will shadow the backup value as well
         if self._test_G.node[nn][self._attr_to_del] == shadow_val:
            continue

         self._shadow_count += 1

         # make a backup and shadow the real val with a shadow val
         self._test_G.node[nn][self._attr_backup]\
               = self._test_G.node[nn][self._attr_to_del]
         self._test_G.node[nn][self._attr_to_del] = shadow_val


class _NodeAttrInferCV(object):

   def _handle_infer(self, subgraph):
      """
         Choose custom or default infer function for 'subgraph',
         and fit the results into a namedtuple.
      """

      res = self._method_instance.infer_attr(subgraph, **self._settings)

      # unshadow inferred values
      if res is not None:
         with_real = []
         for node_id, predicted_val, prediction_base in res:
            if node_id is None:
               continue
            real_val = self._test_G.node[node_id][self._attr_backup]
            with_real.append({ 'node_id': node_id,
                               'real_value': real_val,
                               'predicted_val': predicted_val,
                               'prediction_base': prediction_base })
         return with_real

      return None



   def _gothrough_test_subgraphs(self):
      """
         A generator which yields a subgraph(s) info for which a prediction
         was made.
      """

      index = 1
      for subgraph in self._method_instance.graph_iter(self._test_G,
                                                       **self._settings):
         res = self._handle_infer(subgraph)
         if res:
            print index,
            sys.stdout.flush()
            index += 1
            yield res


   def _infer(self):
      estimates = defaultdict(list)
      done_nodes = []

      for c_data in self._gothrough_test_subgraphs():
         for node_data in c_data:
            nid = node_data['node_id']

            # prevent counting the unshadowing of the same node twice
            if nid in done_nodes:
               continue
            done_nodes.append(nid)

            r_val = node_data['real_value']
            p_val = node_data['predicted_val']

            estimates['node_id'].append(nid)
            estimates['predicted_val'].append(p_val)
            estimates['real_val'].append(r_val)
            estimates['error'].append(p_val - r_val)
            estimates['iteration'].append(self._iteration)
            estimates['prediction_base'].append(node_data['prediction_base'])

      return estimates


   def _graph_check(self):

      sv = self.get_setting('shadow_val')

      for anode in self._G.nodes_iter():

         try:
            assert(self._G.node[anode].has_key(self._attr_to_del))
         except AssertionError:
            raise AssertionError('{} has to be defined for every node! '\
                  'Not defined for node: {}'.format(self._attr_to_del, anode))

         try:
            assert(self._G.node[anode][self._attr_to_del] != sv)
         except AssertionError:
            raise AssertionError('Attribute {} cannot cotain {} values'\
                                 .format(self._attr_to_del, sv))

         try:
            assert(not self._G.node[anode].has_key(self._attr_backup))
         except AssertionError:
            raise AssertionError('Attribute {} already exists for node {}!'\
                                 .format(self._attr_backup, anode))




class NodeAttrInferCV(_NodeAttrInferCV,
                      HandleMethods,
                      HandleSettings,
                      CVGraphPreparation):
   def __init__(self, G_fn=None, method='default'):
      """Test the accuracy of node attribute inference in a graph.

      This class requires a graph not to contain missing data
      (i.e. all attributes ought to contain non-null values).

      Parameters
      ----------

      G_fn : str or None (default: None)
             Path to gml file
             If None, the graph processing functionality is not available

      Attributes
      ----------

      G : networkx.classes.graph.Graph

      test_G : networkx.classes.graph.Graph
               A copy of `G` graph. The original `G` graph is not modified
               by inference accuracy tests. On the other hand, the `test_G`
               is replaced with original `G` graph and then modified in every
               iteration of inference accuracy tests.

      df : pandas.core.frame.DataFrame

      iteration : int


      Methods
      -------

      to_csv(self, csv_fn, attr_to_del, attr_backup, iterations)


      Usage
      -----

      1. Generating csv file with inference accuracy results:

      z = NodeAttrInferCV('example_data/fb_friends_graph.gml')
      z.run()
      z.simple_stats()

      2. Reading csv file:

      z = NodeAttribute_InferenceTest()
      z.from_csv('age_predictions.csv')
      z.error_hist()

      """
      self._G = None
      self._G_fn = G_fn

      self._shadow_count = 0

      self._df = None
      self._iteration = 0
      self._test_G = None

      self._method = method
      self._method_instance = None
      self._methods = available_methods
      self._settings = None
      self.register_method(self._method)

      self._attr_to_del = self.get_setting('attr_to_del')
      self._attr_backup = self.get_setting('attr_backup')


   def _reset(self):
      # these should be set to default before re-run of tests
      self._iteration = 0
      self._shadow_count = 0


   def _run(self):
      for dummy in range(self.get_setting('max_iters')):
         self._iteration += 1
         self._del_random_attributes()
         yield self._infer()


   def run(self, G_fn=None):
      self._reset()
      if self._G is None:
         if self._G_fn is None:
            self._G_fn = G_fn
         if self._G_fn is None: # G_fn still not provided
            print 'No input graph! Exiting...'
            sys.exit()
         self._G = read_gml(self._G_fn)
      self._graph_check()

      # let's collect the estimations for every iteration and store them in df
      estimates = {}
      for est in self._run():
         for k in est:
            estimates[k] = estimates.get(k, []) + est[k]
      self._df = pd.DataFrame(estimates)
      return self._df


   def to_csv(self, csv_fn):
      self._df.to_csv(csv_fn, quoting=QUOTE_ALL, index=False)


   def simple_stats(self):
      shadowed_per_iter = self._shadow_count / self.get_setting('max_iters')
      print 'Shadowed per iteration:', shadowed_per_iter

      df = self._df.copy()
      df['error'] = df['error'].apply(lambda x: 5 if abs(x)>4 else abs(x))
      print 'Unshadowed:', [i for i in df.groupby('iteration').size()]

      ct = pd.crosstab(df['error'], df['iteration'])\
            .apply(lambda r: np.round(r/r.sum(), 2), axis=0)

      print '-'*80

      print ct.ix[0:ct.shape[0]-2]

      print '-'*80

      print ct.ix[0:ct.shape[0]-2].cumsum()

      print '-'*80

      # show biggest prediction errors
      errors_per_iter = defaultdict(list)
      for z in self._df.groupby(['iteration', 'error']).groups.items():
         iterno, error = z[0][0], abs(z[0][1])
         errors_per_iter[iterno].extend([error] * len(z[1]))
      print 'Top 10 biggest errors:'
      for iterno, err_data in errors_per_iter.items():
         print 'Iteration %s: %s.' % (iterno, sorted(err_data)[-10:])


   def error_hist(self, hist_fn=None):
      amin, amax = int(self._df['error'].min()), int(self._df['error'].max())
      self._df['error'].hist(bins=abs(amin)+amax+1)
      if hist_fn:
         plt.savefig(hist_fn)
      else:
         plt.show()


   def from_csv(self, csv_fn):
      self._df = pd.read_csv(csv_fn)
