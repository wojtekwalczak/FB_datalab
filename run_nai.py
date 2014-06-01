import sys
from lib.nai.cv import NodeAttrInferCV

if __name__ == '__main__':
   z = NodeAttrInferCV()

   methods = z.list_methods()

   if len(sys.argv) < 2:
      print 'Usage: %s <%s>' % (sys.argv[0], '|'.join(methods))
      sys.exit(1)

   z.register_method(sys.argv[1])
   z.run('data/fb_friends_graph.gml.gz')
   z.simple_stats()
