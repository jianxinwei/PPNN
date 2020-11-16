#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import copy
import json
import pickle
import sys
import tracemalloc
import time

import ipdb
import memory_profiler
import numpy as np
import pandas as pd
import torch
import torch.distributed.rpc as rpc

sys.path.append('..')
from models.test import test_bank

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

def dump_ip_port_json():
	file = '../ip_port.json'

	ipport_dict = {
		0: {'ip': '127.0.0.1', 'port': 56970},
		1: {'ip': '127.0.0.1', 'port': 56971},
		2: {'ip': '127.0.0.1', 'port': 56972},
		3: {'ip': '127.0.0.1', 'port': 56973},
		4: {'ip': '127.0.0.1', 'port': 56974},
	}

	with open(file, 'w') as f:
		json.dump(ipport_dict, f, indent=4)


def dump_client_server_ip_port_json():
	file = '../ip_port_client_server.json'

	ipport_dict = {
		0: {'ip': '127.0.0.1', 'port': 56970},
		1: {'ip': '127.0.0.1', 'port': 56971},
		2: {'ip': '127.0.0.1', 'port': 56972},
		3: {'ip': '127.0.0.1', 'port': 56973},
		4: {'ip': '127.0.0.1', 'port': 56974},
		5: {'ip': '127.0.0.1', 'port': 56975},
	}

	with open(file, 'w') as f:
		json.dump(ipport_dict, f, indent=4)


def read_ip_port_json(filepath):
	with open(filepath, 'r') as f:
		tmp = json.load(f)
	kv = {}
	for k in tmp.keys():
		kv[int(k)] = tmp[k]
	del tmp
	return kv


# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def normal_train(args, net, optimizer, loss_func, ldr_train, ldr_valid):
        epoch_loss = []
        best_valid_loss = np.finfo(float).max
        best_valid_net = copy.deepcopy(net)

        for iter in range(args.local_ep):
            batch_loss = []
            for batch_idx, (attributes, labels) in enumerate(ldr_train):
                attributes, labels = attributes.to(args.device), labels.to(device=args.device, dtype=torch.long)
                optimizer.zero_grad()
                log_probs = net(attributes)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(attributes), len(ldr_train.dataset),
                              100. * batch_idx / len(ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            net.eval()
            _, tmp_loss_valid = test_bank(net, ldr_valid, args)
            if tmp_loss_valid < best_valid_loss:
            	best_valid_net = copy.deepcopy(net)
            net.train()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)		