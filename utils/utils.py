#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import memory_profiler
import tracemalloc
import time

import torch


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
	tracemalloc.start()
	time0 = time.perf_counter()
	yield
	print('[elapsed time: %.7f s]' % (time.perf_counter() - time0))
	print('[took memory: %.7f Mb]' % (memory_profiler.memory_usage()[0] - m0[0]))
	current, peak = tracemalloc.get_traced_memory()
	print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
