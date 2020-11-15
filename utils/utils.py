#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import tracemalloc
import time

import memory_profiler
import numpy as np
import pandas as pd

import torch

from utils.dataToDataloader import dfToTensor


@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.7f s]' % (time.perf_counter() - time0))


def net_avg_workers(net, num):
	for k in net.keys():
		net[k] = torch.true_divide(net[k], num)
	return net


@contextmanager
def memory_time_moniter():
	"""Helper for measuring runtime and memory cost"""
	m0 = memory_profiler.memory_usage()
	# tracemalloc.start()
	time0 = time.perf_counter()
	yield
	print('[elapsed time: %.7f s]' % (time.perf_counter() - time0))
	print('[took memory: %.7f Mb]' % (memory_profiler.memory_usage()[0] - m0[0]))
	# current, peak = tracemalloc.get_traced_memory()
	# print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")


def data_loading(args):
	if args.dataset != 'farm':
		wholedataset = pd.read_csv('../data/{}/new_{}_whole.csv'.format(args.dataset, args.dataset), sep=';')
		trainset, validset, testset = np.split(wholedataset, [int(args.train_ratio*len(wholedataset)), int((args.train_ratio + args.valid_ratio)*len(wholedataset))])

		train_attributes, train_labels = dfToTensor(trainset)
		train_attributes = train_attributes.to(args.device)
		train_labels = train_labels.to(args.device, dtype=torch.long)
		attrisize = list(train_attributes[0].size())[0]
		classes = args.num_classes

		valid_attributes, valid_labels = dfToTensor(validset)
		valid_attributes = valid_attributes.to(args.device)
		valid_labels = valid_labels.to(args.device, dtype=torch.long)

		test_attributes, test_labels = dfToTensor(testset)
		test_attributes = test_attributes.to(args.device)
		test_labels = test_labels.to(args.device, dtype=torch.long)
	else:
		farm_path = '../data/{}/farm-ads-vect'.format(args.dataset)
		farm_file = open(farm_path)
		lines = farm_file.readlines()
		farm_file.close()
		farm_vector = np.zeros([len(lines),55000])
		farm_label = np.zeros(len(lines))
		for l in range(len(lines)):
			line = lines[l].strip().split(' ')
			farm_label[l] = int(line[0])
			for i in range(1, len(line)):
				index = int(line[i].split(':')[0])
				farm_vector[l][index-1] = 1

		farm_label[farm_label < 0] = 0.0
		farm_label = farm_label.reshape(-1,1)

		farm_data = np.hstack((farm_vector,farm_label))
		wholedataset = pd.DataFrame(farm_data)

		trainset, validset, testset = np.split(wholedataset, [int(args.train_ratio*len(wholedataset)), int((args.train_ratio + args.valid_ratio)*len(wholedataset))])

		train_attributes, train_labels = dfToTensor(trainset)
		train_attributes = train_attributes.to(args.device)
		train_labels = train_labels.to(args.device, dtype=torch.long)
		attrisize = list(train_attributes[0].size())[0]
		classes = args.num_classes

		valid_attributes, valid_labels = dfToTensor(validset)
		valid_attributes = valid_attributes.to(args.device)
		valid_labels = valid_labels.to(args.device, dtype=torch.long)

		test_attributes, test_labels = dfToTensor(testset)
		test_attributes = test_attributes.to(args.device)
		test_labels = test_labels.to(args.device, dtype=torch.long)

	return train_attributes, train_labels, valid_attributes, valid_labels, test_attributes, test_labels