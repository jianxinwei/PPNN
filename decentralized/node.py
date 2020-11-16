#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import copy
import ipdb
import pickle
import select
import socket
import sys

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
			client_sockets[client_sockets_rank2idx[client_rank]].connect((ip_port[client_rank]['ip'], ip_port[client_rank]['port']))
			break
		except:
			print("\33[31m\33[1m Can't connect to the node {} \33[0m".format(client_rank))

def get_listen_connect(args):
	user_list = [idx for idx in range(args.num_users)]
	client_socket_idx = 0
	for client_rank in range(args.num_users):
		if client_rank == args.rank:
			listen()
		else:
			client_sockets_rank2idx[client_rank] = client_socket_idx
			client_sockets_idx2rank[client_socket_idx] = client_rank
			connect(client_rank)
			client_socket_idx += 1

if __name__ == '__main__':
	args = args_parser()
	args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

	save_prefix = '../save/{}_{}_{}_{}_{}_decentralized'.format(args.dataset, args.model, args.optim, args.epochs, args.dataset)

	if args.dp:
		save_prefix = save_prefix + '_dp'
	if args.tphe:
		save_prefix = save_prefix + '_tphe'

	train_attributes, train_labels, valid_attributes, valid_labels, test_attributes, test_labels = data_loading(args) 
	attrisize = list(train_attributes[0].size())[0]
	train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
	valid_loader = DataLoader(dataset=TensorDataset(valid_attributes, valid_labels), batch_size=args.bs, shuffle=True)
	test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)

	ip_port = read_ip_port_json('../ip_port.json')
	self_ip = ip_port[args.rank]['ip']
	self_port = ip_port[args.rank]['port']
	del ip_port[args.rank]
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind((self_ip, self_port))
	server_socket.listen(10) # listen atmost 10 connection at one time
	server_connection_list = []
	client_sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(args.num_users-1)]
	client_sockets_rank2idx = {}
	client_sockets_idx2rank = {}
	client_socket_idx = 0
	for client_rank in range(args.num_users):
		if client_rank == args.rank:
			listen()
		else:
			client_sockets_rank2idx[client_rank] = client_socket_idx
			client_sockets_idx2rank[client_socket_idx] = client_rank
			connect(client_rank)
			client_socket_idx += 1


	for sockfd in client_sockets:
		sockfd.close()
	for sockfd in server_connection_list:
		sockfd.close()
	server_socket.close()
	print('done')