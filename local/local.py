#! ~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import copy
import ipdb
import pickle
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

    save_prefix = '../save/{}_{}_{}_{}_{}_local'.format(args.dataset, args.model, args.optim, args.epochs, args.dataset)

    if args.dp:
        save_prefix = save_prefix + '_dp'
    if args.tphe:
        save_prefix = save_prefix + '_tphe'

    train_attributes, train_labels, valid_attributes, valid_labels, test_attributes, test_labels = data_loading(args) 
    attrisize = list(train_attributes[0].size())[0]
    train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
    valid_loader = DataLoader(dataset=TensorDataset(valid_attributes, valid_labels), batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)

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

    loss_func = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, net_glob)
    if args.dp:
        args.secure_rng = True
        privacy_engine = PrivacyEngine(net_glob, batch_size=args.bs, sample_size=len(train_loader),
                                       alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                                       noise_multiplier=0.3, max_grad_norm=1.2, secure_rng=args.secure_rng)
        privacy_engine.attach(optimizer)
        generator = (
            prng.create_random_device_generator("/dev/urandom") if args.secure_rng else None
        )
        train_loader = prngDataloader(train_attributes, train_labels, [idx for idx in range(len(train_attributes))], batchsize=args.local_bs, gene=generator)

    # training
    loss_train = []
    loss_valid = []
    best_valid_loss = np.finfo(float).max
    best_net_glob = copy.deepcopy(net_glob)

    with memory_time_moniter() as mt:
        for iter in range(args.epochs):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                # data, target = data.to(args.device), target.to(args.device, dtype=torch.long)
                optimizer.zero_grad()
                output = net_glob(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            loss_avg = sum(batch_loss)/len(batch_loss)
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
            net_glob.train()

    torch.save(best_net_glob, save_prefix + '_best.pt')
    torch.save(net_glob, save_prefix + '_final.pt')

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