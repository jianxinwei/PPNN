#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import copy
import multiprocessing

import sys
sys.path.append('..')
from utils.tphe import *

import ipdb

class Worker(object):
	"""docstring for Worker"""
	def __init__(self, update=None, pub_key=None, priv_key=None):
		super(Worker, self).__init__()
		self.update = update
		self.pub_key = pub_key
		self.priv_key = priv_key
		self.tmp_net = None
		self.state_dict = None
		self.tmp_loss = None

	def train(self, net, ldr_train):
		self.state_dict, self.tmp_loss = self.update.train(net, ldr_train)
		# self.tmp_net.load_state_dict(self.state_dict)
		return self.state_dict, self.tmp_loss

	def recv(self, worker_list):
		num = len(worker_list)
		for k in self.state_dict.keys():
			for i in range(num):
				self.state_dict[k] += worker_list[i].state_dict[k]
		return self.export_state_dict(copy.deepcopy(self.state_dict))

	def export_state_dict(self, state_dict):
		return state_dict

	def init_net(self, net):
		self.tmp_net = net

	def update_net(self, broadcast_state_dict, worker_list, total_num):
		num = len(worker_list)
		for k in broadcast_state_dict.keys():
			broadcast_state_dict[k] *= num
			broadcast_state_dict[k] /= (total_num*num)
		self.tmp_net.load_state_dict(broadcast_state_dict)


class TPHEWorker(Worker):
	def __init__(self, update=None, pub_key=None, priv_key=None):
		super().__init__(update, pub_key, priv_key)
		self.encrypted_state_dict = {}
		self.state_shape = {}

	def train(self, net, ldr_train):
		self.state_dict, self.tmp_loss = self.update.train(net, ldr_train)
		# self.tmp_net.load_state_dict(self.state_dict)
		for k in self.state_dict.keys():
			self.encrypted_state_dict[k] = self.state_dict[k].numpy()
			self.state_shape[k] = self.encrypted_state_dict[k].shape
			if len(self.encrypted_state_dict[k].shape) <= 1:
				self.encrypted_state_dict[k] = encrypt_vector(self.pub_key, self.encrypted_state_dict[k])
			else:
				# ravel the high dimension nd array
				self.encrypted_state_dict[k] = encrypt_vector(self.pub_key, self.encrypted_state_dict[k].ravel())
		return self.encrypted_state_dict, self.tmp_loss

	def recv(self, worker_list):
		num = len(worker_list)
		for k in self.encrypted_state_dict.keys():
			for i in range(num):
				self.encrypted_state_dict[k] = batch_add(self.encrypted_state_dict[k], worker_list[i].encrypted_state_dict[k])
		return self.export_encrypted_state_dict(copy.deepcopy(self.encrypted_state_dict))

	def export_encrypted_state_dict(self, encrypted_state_dict):
		return encrypted_state_dict

	def update_net(self, broadcast_encrypted_state_dict, priv_keys, total_num, 
		w, delta, combineSharesConstant, nSPlusOne, n, ns):
		num = len(priv_keys)
		for k in broadcast_encrypted_state_dict.keys():
			shares = [decrypt_vector(priv_key, broadcast_encrypted_state_dict[k]) for priv_key in priv_keys]
			self.state_dict[k] = torch.from_numpy(np.array(batch_decrypt(shares, w, delta, combineSharesConstant, nSPlusOne, n, ns)).reshape(self.state_shape[k])/total_num)
		self.tmp_net.load_state_dict(self.state_dict)