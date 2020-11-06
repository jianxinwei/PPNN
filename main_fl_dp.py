import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
import torchcsprng as prng
from torch.utils.data import DataLoader, TensorDataset
from utils.partition import dataset_iid
from utils.options import args_parser
from utils.dataToDataloader import dfToTensor, prngDataloader
from models.DPUpdate import LocalDPUpdate
from models.Nets import MLP
from models.Fed import FedAvg
from models.test import test_bank
from utils.utils import *

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.all_clients = True
    args.secure_rng = True
    # load dataset and split users
    trainset = pd.read_csv('./data/{}/new_{}_full.csv'.format(args.dataset, args.dataset), sep=';')
    testset = pd.read_csv('./data/{}/new_{}.csv'.format(args.dataset, args.dataset), sep=';')
    train_attributes, train_labels = dfToTensor(trainset)
    attrisize = list(train_attributes[0].size())[0]
    classes = 2
    # print(attrisize, classes)
    test_attributes, test_labels = dfToTensor(testset)
    dict_clients = dataset_iid(trainset, args.num_users)
    test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)

    # build model
    net_glob = MLP(dim_in=attrisize, dim_hidden=args.dim_hidden, dim_out=classes)
    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()
    generator = (
        prng.create_random_device_generator("/dev/urandom") if args.secure_rng else None
    )

    # training
    loss_train = []
    net_best = None
    best_loss = None

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        m = max(int(args.frac * args.num_users), 1)
        users = np.random.choice(range(args.num_users), m, replace=False)
        train_loaders = [prngDataloader(train_attributes, train_labels, dict_clients[u], batchsize=args.local_bs, gene=generator)
            for u in users]
        localss = [LocalDPUpdate(args=args, clientID=u) for u in users]

    with timer() as t:
        for iter in range(args.epochs):
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            for u in users:
                w, loss = localss[u].train(net=copy.deepcopy(net_glob).to(args.device), ldr_train=train_loaders[u])
                if args.all_clients:
                    w_locals[u] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

            # update global weights
            w_glob = FedAvg(w_locals)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round{:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

    torch.save(net_glob, './save/fl_dp_{}.pkl'.format(args.dataset))


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.savefig('./save/{}_fl_dp_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
    # net_glob = torch.load('./save/fl_dp_{}.pkl'.format(args.dataset))
    net_glob.eval()
    acc_train, loss_train = test_bank(net_glob, train_loader, args)
    acc_test, loss_test = test_bank(net_glob, test_loader, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))