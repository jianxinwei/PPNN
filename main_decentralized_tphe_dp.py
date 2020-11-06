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
from models.Worker import *
from utils.tphe import *

import ipdb

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    thresholdPaillier = ThresholdPaillier(args.n_length, args.num_users, args.num_share)
    pub_key = thresholdPaillier.pub_key
    priv_keys = thresholdPaillier.priv_keys


    args.all_clients = True
    # load dataset and split users
    trainset = pd.read_csv('./data/bank/new_bank_full.csv', sep=';')
    testset = pd.read_csv('./data/bank/new_bank.csv', sep=';')
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
        m = max(int(args.frac * args.num_users), 1)
        users = np.random.choice(range(args.num_users), m, replace=False)
        train_loaders = [prngDataloader(train_attributes, train_labels, dict_clients[u], batchsize=args.local_bs, gene=generator)
            for u in users]
        localss = [LocalDPUpdate(args=args, clientID=u) for u in users]
        workers = [TPHEWorker(localss[u], pub_key=pub_key, priv_key=priv_keys[u]) for u in users]
        for u in users:
            workers[u].init_net(copy.deepcopy(net_glob))
  
    with timer() as t:
        for iter in range(args.epochs):
            loss_locals = []
            for u in users:
                _, loss = workers[u].train(net=copy.deepcopy(workers[u].tmp_net).to(args.device), ldr_train=train_loaders[u])
                # _, loss = workers[u].train(net=copy.deepcopy(net_glob).to(args.device), ldr_train=train_loaders[u])
                loss_locals.append(copy.deepcopy(loss))
            cur_mas_id = users[-1]
            broadcast_encrypted_state_dict = workers[cur_mas_id].recv([workers[idx] for idx in users if idx != cur_mas_id])
            for u in users:
                exchange_clients = list(set(list(users)) - set([u]))[:args.num_share]
                priv_keys = [workers[u_idx].priv_key for u_idx in exchange_clients]
                workers[u].update_net(copy.deepcopy(broadcast_encrypted_state_dict), priv_keys, args.num_users,
                     thresholdPaillier.w, thresholdPaillier.delta, thresholdPaillier.combineSharesConstant, 
                     pub_key.nSPlusOne, pub_key.n, pub_key.ns)
            # net_glob.load_state_dict(net_avg_workers(broadcast_state_dict, args.num_users))
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round{:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

    # net_glob.load_state_dict(net_avg_workers(broadcast_state_dict, args.num_users))
    net_glob = workers[0].tmp_net # Randomly copy the weight from a client. Here, we choose client 0.
    torch.save(net_glob, './save/decentralized_tphe_dp_bank.pkl')

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.savefig('./save/bank_decentralized_tphe_dp_{}_{}.png'.format(args.model, args.epochs))

    # testing
    train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
    # net_glob = torch.load('./save/decentralized_tphe_dp_bank.pkl')
    net_glob.eval()
    acc_train, loss_train = test_bank(net_glob, train_loader, args)
    acc_test, loss_test = test_bank(net_glob, test_loader, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))