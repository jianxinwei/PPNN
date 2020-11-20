#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import copy
import ipdb
import pickle
import random
import select
import socket
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from opacus import PrivacyEngine
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchcsprng as prng

sys.path.append('..')

from utils.partition import dataset_iid
from utils.options import args_parser
from models.Nets import *
from models.Fed import FedAvg
from models.test import test_bank
from utils.data_to_dataloader import *
from utils.optim import *
from utils.tphe import *
from utils.utils import *


def listen():
	while True:
		sockfd, addr = server_socket.accept()
		server_connection_list.append(sockfd)
		if len(server_connection_list) == args.num_users-1:
			break


def connect(client_rank):
	while True:
		try:
			client_sockets[rank2idx[client_rank]].connect((ip_port[client_rank]['ip'], ip_port[client_rank]['port']))
			client_sockets[rank2idx[client_rank]].setblocking(True)
			break
		except:
			# print("\33[31m\33[1m Can't connect to the node {} \33[0m".format(client_rank))
			pass
		time.sleep(1)


def send_net_to_all(net):
	for sockfd in server_connection_list:
		sockfd.send(pickle.dumps(copy.deepcopy(net)))


def send_aggregated_weight_state_dict_to_all(aggregated_state_dict):
	for sockfd in server_connection_list:
		sockfd.send(pickle.dumps(copy.deepcopy(aggregated_state_dict)))


def send_metadata_to_all(cur_train_loss, cur_valid_loss, next_server_rank_id):
	for sockfd in server_connection_list:
		sockfd.send(pickle.dumps([cur_train_loss, cur_valid_loss, next_server_rank_id]))


def send_tphe_to_all():
	for idx, sockfd in enumerate(server_connection_list):
		# paritial private key shares set up for server node
		rank_sampling_list = [rank for rank in range(args.num_users) if rank != idx2rank[idx]]
		random.shuffle(rank_sampling_list)
		tmp_partial_priv_keys = []
		tmp_partial_priv_keys.append(priv_keys[idx2rank[idx]])
		for rank_idx in rank_sampling_list[:args.num_share-1]:
			tmp_partial_priv_keys.append(priv_keys[rank_idx])
		assert len(tmp_partial_priv_keys) == args.num_share
		sockfd.send(pickle.dumps([pub_key, priv_keys[idx2rank[idx]], tp_w, tp_delta, tp_combineSharesConstant, tmp_partial_priv_keys]))
		del tmp_partial_priv_keys

def state_dict_aggregation(w_state_dict_locals):
	intermediate_state_dict = copy.deepcopy(w_state_dict_locals[0])
	if args.tphe:
		for k in net_glob_state_dict.keys():
			for idx in range(1, len(w_state_dict_locals)):
				intermediate_state_dict[k] = batch_add(intermediate_state_dict[k], w_state_dict_locals[idx][k])
	else:
		intermediate_state_dict = FedAvg(w_state_dict_locals)
	return intermediate_state_dict


def parse_aggregated_state_dict(aggregated_state_dict, cur_net):
	if args.tphe:
		decrypted_state_dict = decrypt_torch_state_dict(aggregated_state_dict, partial_priv_keys, 
			tp_w, tp_delta, tp_combineSharesConstant, pub_key.nSPlusOne, pub_key.n, pub_key.ns, args.num_users, cur_net.state_dict())
	else:
		cur_net.load_state_dict(aggregated_state_dict)


def server(cur_net, current_iter, current_server_rank_id, best_valid_loss, best_net_glob, server_flag):
	loss_locals = []
	w_state_dict_locals = []

	# local train
	cur_net.train()
	optimizer = get_optimizer(args, cur_net)
	loss_func = nn.CrossEntropyLoss()
	if args.dp:
		privacy_engine = PrivacyEngine(cur_net, batch_size=args.bs, sample_size=len(local_train_loader),
										alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
										noise_multiplier=0.3, max_grad_norm=1.2, secure_rng=args.secure_rng)
		privacy_engine.attach(optimizer)
	current_state_dict, current_loss = normal_train(args, cur_net, optimizer, loss_func, local_train_loader, valid_loader)

	if args.dp:
		privacy_engine.detach()

	loss_locals.append(current_loss)
	if args.tphe:
		w_state_dict_locals.append(encrypt_torch_state_dict(pub_key, current_state_dict))
	else:
		w_state_dict_locals.append(current_state_dict)

	# receive from others
	loop = True
	while loop:
		# Get the list sockets which are ready to be read through select
		rList, wList, error_sockets = select.select(server_connection_list,[],[])
		for sockfd in rList:
			tmp_pkl_data = sockfd.recv(int(args.buffer)) # 760586945
			tmp_state_dict, tmp_loss = pickle.loads(tmp_pkl_data)
			w_state_dict_locals.append(tmp_state_dict)
			loss_locals.append(tmp_loss)
			if len(w_state_dict_locals) == args.num_users:
				loop = False
				break

	# aggregate weight state_dicts
	aggregated_state_dict = state_dict_aggregation(w_state_dict_locals)

	# distribute the aggregated weight state_dict
	send_aggregated_weight_state_dict_to_all(aggregated_state_dict)

	# parse aggregated state_dict
	parse_aggregated_state_dict(aggregated_state_dict, cur_net)

	loss_avg = sum(loss_locals) / len(loss_locals)
	print('Round{:3d}, Average loss {:.3f}'.format(current_iter, loss_avg))
	loss_train.append(loss_avg)
	cur_net.eval()
	acc_valid, tmp_loss_valid = test_bank(cur_net, valid_loader, args)
	print('Round{:3d}, Validation loss {:.3f}'.format(current_iter, tmp_loss_valid))
	loss_valid.append(tmp_loss_valid)
	if tmp_loss_valid < best_valid_loss:
		best_valid_loss = tmp_loss_valid
		best_net_glob = copy.deepcopy(cur_net)
		print('SAVE BEST MODEL AT EPOCH {}'.format(current_iter))

	# pick the server for next epoch
	next_server_rank_id = random.randint(0, args.num_users-1)

	# distribute metadata
	send_metadata_to_all(loss_avg, tmp_loss_valid, next_server_rank_id)

	if next_server_rank_id != args.rank:
		server_flag = False
		current_server_rank_id = next_server_rank_id

	print("\33[31m\33[1m Current server rank id {} \33[0m".format(current_server_rank_id))

	return cur_net, current_server_rank_id, best_valid_loss, best_net_glob, server_flag


def client(cur_net, current_iter, current_server_rank_id, best_valid_loss, best_net_glob, server_flag):
	# local train
	cur_net.train()
	optimizer = get_optimizer(args, cur_net)
	loss_func = nn.CrossEntropyLoss()
	if args.dp:
		privacy_engine = PrivacyEngine(cur_net, batch_size=args.bs, sample_size=len(local_train_loader),
										alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
										noise_multiplier=0.3, max_grad_norm=1.2, secure_rng=args.secure_rng)
		privacy_engine.attach(optimizer)
	current_state_dict, current_loss = normal_train(args, cur_net, optimizer, loss_func, local_train_loader, valid_loader)

	if args.dp:
		privacy_engine.detach()

	# send the state_dict to current server
	if args.tphe:
		client_sockets[rank2idx[current_server_rank_id]].send(pickle.dumps([encrypt_torch_state_dict(pub_key, current_state_dict), current_loss]))
	else:
		client_sockets[rank2idx[current_server_rank_id]].send(pickle.dumps([current_state_dict, current_loss]))

	# recv the aggregated state dict from current server
	aggregated_state_dict = client_sockets[rank2idx[current_server_rank_id]].recv(int(args.buffer))
	aggregated_state_dict = pickle.loads(aggregated_state_dict)

	# parse aggregated state_dict
	parse_aggregated_state_dict(aggregated_state_dict, cur_net)

	# recv metadata
	metadata_list_pkl = client_sockets[rank2idx[current_server_rank_id]].recv(int(args.buffer))
	loss_avg, tmp_loss_valid, next_server_rank_id = pickle.loads(metadata_list_pkl)
	loss_train.append(loss_avg)
	loss_valid.append(tmp_loss_valid)
	print('Round{:3d}, Average loss {:.3f}'.format(current_iter, loss_avg))
	print('Round{:3d}, Validation loss {:.3f}'.format(current_iter, tmp_loss_valid))
	if tmp_loss_valid < best_valid_loss:
		best_valid_loss = tmp_loss_valid
		best_net_glob = copy.deepcopy(cur_net)
		print('SAVE BEST MODEL AT EPOCH {}'.format(current_iter))

	# update the metadata for server
	current_server_rank_id = next_server_rank_id
	if next_server_rank_id == args.rank:
		server_flag = True

	print("\33[31m\33[1m Current server rank id {} \33[0m".format(current_server_rank_id))

	return cur_net, current_server_rank_id, best_valid_loss, best_net_glob, server_flag


if __name__ == '__main__':
	args = args_parser()
	args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

	# rank id should start from 0
	# the default server for the first epoch is rank id 0
	if args.rank == None:
		sys.exit(1)

	save_prefix = '../save/{}_{}_{}_{}_decentralized'.format(args.dataset, args.model, args.optim, args.epochs)
	if args.dp:
		save_prefix = save_prefix + '_dp'
	if args.tphe:
		save_prefix = save_prefix + '_tphe'

	current_server_rank_id = 0
	server_flag = False
	if args.rank == 0:
		server_flag = True

	# TPHE set up
	thresholdPaillier = None
	pub_key = None
	priv_key = None
	priv_keys = None
	tp_w = None
	tp_delta = None
	tp_combineSharesConstant = None
	partial_priv_keys = None
	if args.tphe and server_flag:
		thresholdPaillier = ThresholdPaillier(args.n_length, args.num_users, args.num_share)
		pub_key = thresholdPaillier.pub_key
		priv_keys = thresholdPaillier.priv_keys
		priv_key = priv_keys[args.rank]
		tp_w = thresholdPaillier.w
		tp_delta = thresholdPaillier.delta
		tp_combineSharesConstant = thresholdPaillier.combineSharesConstant

		# paritial private key shares set up for server node
		rank_sampling_list = [rank for rank in range(args.num_users) if rank != args.rank]
		random.shuffle(rank_sampling_list)
		partial_priv_keys = []
		partial_priv_keys.append(priv_key)
		for rank_idx in rank_sampling_list[:args.num_share-1]:
			partial_priv_keys.append(priv_keys[rank_idx])
		assert len(partial_priv_keys) == args.num_share

	train_attributes, train_labels, valid_attributes, valid_labels, test_attributes, test_labels = data_loading(args) 
	attrisize = list(train_attributes[0].size())[0]
	train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
	valid_loader = DataLoader(dataset=TensorDataset(valid_attributes, valid_labels), batch_size=args.bs, shuffle=True)
	test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)

	local_train_idxes = [idx for idx in range(int(train_attributes.shape[0]*args.rank/args.num_users),int(train_attributes.shape[0]*(args.rank+1)/args.num_users))]
	local_train_loader = clientDataloader(train_attributes, train_labels, local_train_idxes, batchsize=args.local_bs)
	if args.dp:
		args.secure_rng = True
		generator = (prng.create_random_device_generator("/dev/urandom") if args.secure_rng else None)
		local_train_loader = prngDataloader(train_attributes, train_labels, local_train_idxes, batchsize=args.local_bs, gene=generator)

	# Initialize socket connections
	# ip_port = read_ip_port_json('../ip_port_client_server.json')
	json_path = '../json/decentralized_{}_{}.json'.format(args.dataset, args.optim.lower())
	if args.dp:
		json_path = '../json/decentralized_{}_{}_dp.json'.format(args.dataset, args.optim.lower())
	if args.tphe:
		json_path = '../json/decentralized_{}_{}_tphe.json'.format(args.dataset, args.optim.lower())
		if args.dp:
			json_path = '../json/decentralized_{}_{}_dp_tphe.json'.format(args.dataset, args.optim.lower())
	ip_port = read_ip_port_json(json_path)
	# print(ip_port)

	self_ip = ip_port[args.rank]['ip']
	self_port = ip_port[args.rank]['port']
	del ip_port[args.rank]
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind((self_ip, self_port))
	server_socket.listen(10) # listen atmost 10 connection at one time
	server_connection_list = []
	client_sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(args.num_users-1)]
	rank2idx = {}
	idx2rank = {}
	client_socket_idx = 0
	for client_rank in range(args.num_users):
		if client_rank == args.rank:
			listen()
		else:
			rank2idx[client_rank] = client_socket_idx
			idx2rank[client_socket_idx] = client_rank
			connect(client_rank)
			client_socket_idx += 1

	# build model
	if args.gpu != -1:
		net_glob = MLP(dim_in=attrisize, dim_hidden=args.dim_hidden, dim_out=args.num_classes).to(args.device)
	else:
		net_glob = MLP(dim_in=attrisize, dim_hidden=args.dim_hidden, dim_out=args.num_classes)
	if args.dataset == 'farm':
		if args.gpu != -1:
			net_glob = LargeMLP(dim_in=attrisize, args=args, dim_out=args.num_classes).to(args.device)
		else:
			net_glob = LargeMLP(dim_in=attrisize, args=args, dim_out=args.num_classes)
	print(net_glob)
	net_glob_state_dict = net_glob.state_dict()

	# training
	loss_train = []
	loss_valid = []
	best_valid_loss = np.finfo(float).max
	best_net_glob = None


	if server_flag: 
		send_net_to_all(net_glob) # distribute the initial net to others
		if args.tphe:
			send_tphe_to_all() # distribute tphe elements to others
	else: 
		# receive the initiali net from current server
		tmp_pkl_data = client_sockets[rank2idx[current_server_rank_id]].recv(int(args.buffer))
		net_glob = pickle.loads(tmp_pkl_data)
		if args.tphe:
			# receive the tphe key pair
			pkl_list = client_sockets[rank2idx[current_server_rank_id]].recv(int(args.buffer))
			pub_key, priv_key, tp_w, tp_delta, tp_combineSharesConstant, partial_priv_keys = pickle.loads(pkl_list)
	
	with memory_time_moniter() as mt:
		for iter in range(args.epochs):
			if server_flag:
				net_glob, current_server_rank_id, best_valid_loss, best_net_glob, server_flag = server(net_glob, iter, 
					current_server_rank_id, best_valid_loss, best_net_glob, server_flag)
				if iter == args.epochs-1: # Prevent other nodes' write to the same file
					torch.save(best_net_glob, save_prefix + '_best.pt')
					torch.save(net_glob, save_prefix + '_final.pt')
			else:
				net_glob, current_server_rank_id, best_valid_loss, best_net_glob, server_flag = client(net_glob, iter, 
					current_server_rank_id, best_valid_loss, best_net_glob, server_flag)

	# close all sockets
	for sockfd in client_sockets:
		sockfd.close()
	for sockfd in server_connection_list:
		sockfd.close()
	server_socket.close()

	# plot loss curve
	plt.figure()
	plt.plot(range(len(loss_train)), loss_train, 'r', label='train_loss')
	plt.plot(range(len(loss_valid)), loss_valid, 'b', label='valid_loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.grid(True)
	plt.legend(loc=0)
	plt.savefig(save_prefix + '.png')

	# testing
	net_glob = torch.load(save_prefix + '_best.pt')
	net_glob.eval()
	acc_train, loss_train = test_bank(net_glob, train_loader, args)
	acc_valid, loss_valid = test_bank(net_glob, valid_loader, args)
	acc_test, loss_test = test_bank(net_glob, test_loader, args)
	print("Training accuracy: {:.2f}".format(acc_train), "Training loss: {:.2f}".format(loss_train))
	print("Validating accuracy: {:.2f}".format(acc_valid), "Validating loss: {:.2f}".format(loss_valid))
	print("Testing accuracy: {:.2f}".format(acc_test), "Testing loss: {:.2f}".format(loss_test))