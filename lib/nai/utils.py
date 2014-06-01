# -*- coding: utf-8 -*-
"""
   nai.utils.py
   ~~~~~~~~~~~~

"""

import networkx as nx
from statlib.stats import mean, stdev
from multiprocessing import Process, Queue
from collections import defaultdict, namedtuple


def mean_stdev(vals, null_val=-1):
   vals2 = [i for i in vals if i != null_val]
   return int(round(mean(vals2), 0)), stdev(vals2), len(vals2)

def vals_by_key(adict, akey):
   return [i[akey] for i in adict.values()]

def vals_by_key_filter(adict, akey, filter_val):
   return [i[akey] for i in adict.values() if i[akey] != filter_val]


def add_node_attrs(aclique, source_G):
   with_attrs = defaultdict(dict)

   for anode in aclique:
      for nattr in source_G.node[anode].keys(): #node_attrs:
         if nattr in ('id', 'label'):
            continue
         with_attrs[anode][nattr] = source_G.node[anode][nattr]
   return with_attrs


def read_gml(G_fn):
   """Run networkx's read_gml() function as a separate process to ensure
   that the memory used for parsing the source file will be released
   at return."""
   def do_read(G_fn, q):
      q.put(nx.read_gml(G_fn))

   q = Queue()
   p = Process(target=do_read, args=(G_fn, q))
   p.start()
   result = q.get()
   if p.is_alive():
      p.terminate()
   return result

