#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import json

import ipdb

file = 'ipport.json'

ipport_dict = {
	0: {'ip': '127.0.0.1', 'port': 56970},
	1: {'ip': '127.0.0.1', 'port': 56971},
	2: {'ip': '127.0.0.1', 'port': 56972},
	3: {'ip': '127.0.0.1', 'port': 56973},
	4: {'ip': '127.0.0.1', 'port': 56974},
}

with open(file, 'w') as f:
	json.dump(ipport_dict, f, indent=4)