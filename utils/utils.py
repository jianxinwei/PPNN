#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
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