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


def send_net_to_all(net):
	for sockfd in connected_list:
		sockfd.send(pickle.dumps(copy.deepcopy(net)))


def send_terminate_to_all():
	for sockfd in connected_list:
		sockfd.send(pickle.dumps("terminate"))


if __name__ == '__main__':
	args = args_parser()
	args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

	save_prefix = '../save/{}_{}_{}_{}_{}_client_server'.format(args.dataset, args.model, args.optim, args.epochs, args.dataset)

	if args.dp:
		save_prefix = save_prefix + '_dp'
	if args.tphe:
		save_prefix = save_prefix + '_tphe'

	train_attributes, train_labels, valid_attributes, valid_labels, test_attributes, test_labels = data_loading(args) 
	attrisize = list(train_attributes[0].size())[0]
	train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
	valid_loader = DataLoader(dataset=TensorDataset(valid_attributes, valid_labels), batch_size=args.bs, shuffle=True)
	test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)

	ip_port = read_ip_port_json('../ip_port_client_server.json')
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind((ip_port[0]['ip'], ip_port[0]['port'])) # default server rank id: 0
	server_socket.listen(10) # listen atmost 10 connection at one time
	# server_socket.setblocking(True) # blocking mode
	connected_list = []
	print("\33[32m \t\t\t\tSERVER WORKING \33[0m")

	# training
	loss_train = []
	loss_valid = []
	best_valid_loss = np.finfo(float).max
	best_net_glob = None

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
	net_glob.train()

	while True:
		sockfd, addr = server_socket.accept()
		connected_list.append(sockfd)
		if len(connected_list) == args.num_users:
			break

	with memory_time_moniter() as mt:
		for iter in range(args.epochs):
			loss_locals = []
			w_state_dict_locals = []
			send_net_to_all(net_glob)
			loop = True
			while loop:
				# Get the list sockets which are ready to be read through select
				rList, wList, error_sockets = select.select(connected_list,[],[])
				for sockfd in rList:
					tmp_pkl_data = sockfd.recv(int(args.buffer)) # 760586945
					tmp_state_dict, tmp_loss = pickle.loads(tmp_pkl_data)
					w_state_dict_locals.append(tmp_state_dict)
					loss_locals.append(tmp_loss)
					if len(w_state_dict_locals) == args.num_users:
						loop = False
						break

			# update global weights
			w_state_dict_glob = FedAvg(w_state_dict_locals)
			# copy weight to net_glob
			net_glob.load_state_dict(w_state_dict_glob)
			loss_avg = sum(loss_locals) / len(loss_locals)
			print('Round{:3d}, Average loss {:.3f}'.format(iter, loss_avg))
			loss_train.append(loss_avg)

			net_glob.eval()
			acc_valid, tmp_loss_valid = test_bank(net_glob, valid_loader, args)
			print('Round{:3d}, Validation loss {:.3f}'.format(iter, tmp_loss_valid))
			loss_valid.append(tmp_loss_valid)
			if tmp_loss_valid < best_valid_loss:
				best_valid_loss = tmp_loss_valid
				best_net_glob = copy.deepcopy(net_glob)
				print('SAVE BEST MODEL AT EPOCH {}'.format(iter))

	torch.save(best_net_glob, save_prefix + '_best.pt')
	torch.save(net_glob, save_prefix + '_final.pt')
	send_terminate_to_all()

	for sockfd in connected_list:
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