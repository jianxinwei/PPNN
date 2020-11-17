#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
	parser = argparse.ArgumentParser()
	# federated arguments
	parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
	# parser.add_argument('--epochs', type=int, default=3, help="rounds of training")
	parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
	# parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
	parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
	parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
	# parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
	parser.add_argument('--local_bs', type=int, default=8, help="local batch size: B")
	parser.add_argument('--bs', type=int, default=8, help="test batch size")
	parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
	parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
	parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

	# model arguments
	parser.add_argument('--model', type=str, default='mlp', help='model name')
	parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
	parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
						help='comma-separated kernel size to use for convolution')
	parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
	parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
	parser.add_argument('--max_pool', type=str, default='True',
						help="Whether use max pooling rather than strided convolutions")
	parser.add_argument("--delta", type=float, default=1e-5, metavar="D", help="Target delta (default: 1e-5)")
	parser.add_argument("--sigma", type=float, default=1.0, metavar="S", help="Noise multiplier (default 1.0)")

	# other arguments
	parser.add_argument('--dataset', type=str, default='bank', help="name of dataset")
	parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
	parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
	parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
	parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
	parser.add_argument('--verbose', action='store_true', help='verbose print')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	parser.add_argument('--all_clients', action='store_true', default=False, help='aggregation over all clients')
	parser.add_argument('--secure_rng', action='store_true', default=False,
						help='Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost')
	parser.add_argument('--pb_interval', type=int, default=200, help='Computes the (epsilon, delta) privacy budget spent so far.')
	parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
	parser.add_argument('--dim_hidden', type=int, default=32, help='hidden layer dimension')
	parser.add_argument('--n_length', type=int, default=1024, help='Key size in bits, designed for TPHE')
	parser.add_argument('--num_share', type=int, default=3, help='Minimum number to decrpt ciphertexts, designed for TPHE')
	
	parser.add_argument('--train_ratio', type=float, default=0.6)
	parser.add_argument('--valid_ratio', type=float, default=0.2)
	parser.add_argument('--test_ratio', type=float, default=0.2)

	parser.add_argument('--optim', type=str, default='sgd')

	parser.add_argument('--dp', action='store_true', help='Enable differential privacy')
	parser.add_argument('--tphe', action='store_true', help='Enable TPHE')

	parser.add_argument('--buffer', type=float, default=50e8) # 760586945 * 5

	parser.add_argument('--rank', type=int, default=None)


	parser.add_argument('--farm_hidden_1', type=int, default=3430)
	parser.add_argument('--farm_hidden_2', type=int, default=428)
	parser.add_argument('--farm_hidden_3', type=int, default=54)
	parser.add_argument('--farm_hidden_4', type=int, default=8)

	# parser.add_argument('--farm_hidden_0', type=int, default=13720)
	# parser.add_argument('--farm_hidden_1', type=int, default=3430)
	# parser.add_argument('--farm_hidden_2', type=int, default=858)
	# parser.add_argument('--farm_hidden_3', type=int, default=214)
	# parser.add_argument('--farm_hidden_4', type=int, default=54)
	# parser.add_argument('--farm_hidden_5', type=int, default=14)
	# parser.add_argument('--farm_hidden_6', type=int, default=4)

	args = parser.parse_args()
	return args