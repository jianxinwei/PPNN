#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import json
import os
import random

json_dir = '../json'

dataset_list = ['bank', 'bank_random', 'bidding', 'credit', 'farm']
optim_list = ['sgd', 'adam', 'adafactor', 'novograd']

start_port = random.randint(50000, 60000)
port_cursor = 0


# client server port set up
for dataset in dataset_list:
	for optim in optim_list:
		json_path = os.path.join(json_dir, 'client_server_' + dataset + '_' + optim + '.json')
		ipport_dict = {}
		for idx in range(6):
			ipport_dict[idx] = {'ip': '127.0.0.1', 'port': start_port + port_cursor}
			port_cursor += 1
		with open(json_path, 'w') as f:
			json.dump(ipport_dict, f, indent=4)
		json_path = os.path.join(json_dir, 'client_server_' + dataset + '_' + optim + '_dp.json')
		ipport_dict = {}
		for idx in range(6):
			ipport_dict[idx] = {'ip': '127.0.0.1', 'port': start_port + port_cursor}
			port_cursor += 1
		with open(json_path, 'w') as f:
			json.dump(ipport_dict, f, indent=4)

# decentralized port set up
for dataset in dataset_list:
	for optim in optim_list:
		json_path = os.path.join(json_dir, 'decentralized_' + dataset + '_' + optim + '.json')
		ipport_dict = {}
		for idx in range(5):
			ipport_dict[idx] = {'ip': '127.0.0.1', 'port': start_port + port_cursor}
			port_cursor += 1
		with open(json_path, 'w') as f:
			json.dump(ipport_dict, f, indent=4)
		json_path = os.path.join(json_dir, 'decentralized_' + dataset + '_' + optim + '_dp.json')
		ipport_dict = {}
		for idx in range(5):
			ipport_dict[idx] = {'ip': '127.0.0.1', 'port': start_port + port_cursor}
			port_cursor += 1
		with open(json_path, 'w') as f:
			json.dump(ipport_dict, f, indent=4)
		json_path = os.path.join(json_dir, 'decentralized_' + dataset + '_' + optim + '_tphe.json')
		ipport_dict = {}
		for idx in range(5):
			ipport_dict[idx] = {'ip': '127.0.0.1', 'port': start_port + port_cursor}
			port_cursor += 1
		with open(json_path, 'w') as f:
			json.dump(ipport_dict, f, indent=4)
		json_path = os.path.join(json_dir, 'decentralized_' + dataset + '_' + optim + '_dp_tphe.json')
		ipport_dict = {}
		for idx in range(5):
			ipport_dict[idx] = {'ip': '127.0.0.1', 'port': start_port + port_cursor}
			port_cursor += 1
		with open(json_path, 'w') as f:
			json.dump(ipport_dict, f, indent=4)