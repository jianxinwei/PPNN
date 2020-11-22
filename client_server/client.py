#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import copy
import ipdb
import pickle
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

if __name__ == '__main__':
	args = args_parser()
	args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

	# rank should start from 1, for the server rank id is 0
	if args.rank == None:
		sys.exit(1)

	ip_port = read_ip_port_json('../ip_port_client_server.json')
	# json_path = '../json/client_server_{}_{}.json'.format(args.dataset, args.optim.lower())
	# if args.dp:
		# json_path = '../json/client_server_{}_{}_dp.json'.format(args.dataset, args.optim.lower())
	# ip_port = read_ip_port_json(json_path)

	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	server_socket.setblocking(True) # blocking mode

	try:
		server_socket.connect((ip_port[0]['ip'], ip_port[0]['port']))
	except:
		print("\33[31m\33[1m Can't connect to the server \33[0m")
		sys.exit(1)

	train_attributes, train_labels, valid_attributes, valid_labels, test_attributes, test_labels = data_loading(args) 
	attrisize = list(train_attributes[0].size())[0]
	valid_loader = DataLoader(dataset=TensorDataset(valid_attributes, valid_labels), batch_size=args.bs, shuffle=True)
	test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)
	local_train_idxes = [idx for idx in range(int(train_attributes.shape[0]*(args.rank-1)/args.num_users),int(train_attributes.shape[0]*args.rank/args.num_users))]
	local_train_loader = clientDataloader(train_attributes, train_labels, local_train_idxes, batchsize=args.local_bs)
	if args.dp:
		args.secure_rng = True
		generator = (prng.create_random_device_generator("/dev/urandom") if args.secure_rng else None)
		local_train_loader = prngDataloader(train_attributes, train_labels, local_train_idxes, batchsize=args.local_bs, gene=generator)

	# load model from server
	with memory_time_moniter() as mt:
		while True:
			pkl_data = server_socket.recv(int(args.buffer)) # 760586945
			tmp_data = pickle.loads(pkl_data)
			if tmp_data == 'terminate':
				break
			else:
				tmp_data.train()
				optimizer = get_optimizer(args, tmp_data)
				loss_func = nn.CrossEntropyLoss()
				if args.dp:
					privacy_engine = PrivacyEngine(tmp_data, batch_size=args.bs, sample_size=len(local_train_loader),
													alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
													noise_multiplier=0.3, max_grad_norm=1.2, secure_rng=args.secure_rng)
					privacy_engine.attach(optimizer)
				current_state_dict, current_loss = normal_train(args, tmp_data, optimizer, loss_func, local_train_loader, valid_loader)
				if args.dp:
					privacy_engine.detach()
				server_socket.send(pickle.dumps([current_state_dict, current_loss]))
	server_socket.close()