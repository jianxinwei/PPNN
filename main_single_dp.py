import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils.partition import dataset_iid
from utils.options import args_parser
from utils.dataToDataloader import dfToTensor, clientDataloader
from models.Nets import MLP
from models.Fed import FedAvg
from models.test import test_bank
from opacus import PrivacyEngine
from utils.utils import *

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trainset = pd.read_csv('./data/bank/new_bank_full.csv', sep=';')
    testset = pd.read_csv('./data/bank/new_bank.csv', sep=';')
    train_attributes, train_labels = dfToTensor(trainset)
    train_attributes = train_attributes.to(args.device)
    train_labels = train_labels.to(args.device, dtype=torch.long)
    attrisize = list(train_attributes[0].size())[0]
    classes = 2

    # print(attrisize, classes)
    test_attributes, test_labels = dfToTensor(testset)
    test_attributes = test_attributes.to(args.device)
    test_labels = test_labels.to(args.device, dtype=torch.long)
    # dict_clients = dataset_iid(trainset, args.num_users)
    test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True, drop_last=True)

    # build model
    if args.gpu != -1:
        net_glob = MLP(dim_in=attrisize, dim_hidden=args.dim_hidden, dim_out=classes).to(args.device)
    else:
        net_glob = MLP(dim_in=attrisize, dim_hidden=args.dim_hidden, dim_out=classes)

    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    net_best = None
    best_loss = None

    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
    loss_func = nn.CrossEntropyLoss()

    args.secure_rng = True
    privacy_engine = PrivacyEngine(net_glob, batch_size=args.bs, sample_size=len(train_loader),
                                   alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                                   noise_multiplier=0.3, max_grad_norm=1.2, secure_rng=args.secure_rng)
    privacy_engine.attach(optimizer)


    with timer() as t:
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

            if batch_idx % args.pb_interval == 0:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
                print(
                    f"\tTrain Epoch: {iter} \t"
                    f"Loss: {loss_avg:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta})"
                )

    torch.save(net_glob, './save/single_dp_bank.pkl')

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.savefig('./save/bank_single_dp_{}_{}.png'.format(args.model, args.epochs))

    # testing
    # net_glob = torch.load('./save/single_dp_bank.pkl')
    net_glob.eval()
    if args.gpu != -1:
        test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)
    acc_train, loss_train = test_bank(net_glob, train_loader, args)
    acc_test, loss_test = test_bank(net_glob, test_loader, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))