#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os
import sys

dataset_id = int(sys.argv[1])

if dataset_id == 0:
	dataset_name = 'bank'
elif dataset_id == 1:
	dataset_name = 'bank_random'
elif dataset_id == 2:
	dataset_name = 'bidding'
else:
	dataset_name = 'credit'

log_path = os.path.join('../..', 'logs')


def read_test_acc(filename):
	test_acc = 0.0
	with open(filename, 'r') as f:
		while True:
		 	line = f.readline()
		 	if not line:
		 		break
		 	if line.startswith('Testing'):
		 		line = line.strip().split(' ')
		 		test_acc = float(line[2])
	return test_acc


if __name__ == '__main__':
	model_names = ['single_', 'single_dp_',
		'one_node_', 'one_node_dp_',
		'fl_', 'fl_dp_', 
		'decentralized_', 'decentralized_dp_',
		'decentralized_tphe_', 'decentralized_tphe_dp_']

	for model_name in model_names:
		print(model_name[:-1], read_test_acc(os.path.join(log_path, model_name + dataset_name + '.log')))