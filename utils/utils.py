#!/usr/bin/env python
# -*- encofing: utf-8

import gzip
import msgpack

class Utils(object):
   def _load_pickle(self, fn):
      w = gzip.open(fn)
      data = msgpack.unpack(w)
      w.close()
      return data
