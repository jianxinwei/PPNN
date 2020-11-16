from node import *
from nodeconnection import *
import sys
import time

cur_node = Node("localhost", 56561)

cur_node.start()
cur_node.debug = True

while True:
	cur_node.connect_with_node("localhost", 56562)
	cur_node.connect_with_node("localhost", 56563)
	time.sleep(3)
	cur_node.print_connections()
	if len(cur_node.nodes_inbound) == 2 and len(cur_node.nodes_outbound) == 2:
		break

print('OUTSIDE !!!')
print(cur_node.all_nodes)
cur_node.stop()