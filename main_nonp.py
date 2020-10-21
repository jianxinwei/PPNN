import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.partition import dataset_iid
from utils.options import args_parser
from models.Update import LocalUpdate, dfToTensor, clientDataloader
from models.Nets import MLP
from models.Fed import FedAvg
from models.test import test_bank

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cpu')
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
    test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs,
                             shuffle=True)

    # build model
    net_glob = MLP(dim_in=attrisize, dim_hidden=32, dim_out=classes)
    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    net_best = None
    best_loss = None

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        users = np.random.choice(range(args.num_users), m, replace=False)
        for u in users:
            train_loader = clientDataloader(train_attributes, train_labels, dict_clients[0],
                                            batchsize=args.local_bs)  # args.batch_size
            local = LocalUpdate(args=args, train_loader=train_loader)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
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

    # torch.save(net_glob, './models/net_nonp_bank.pkl')

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.savefig('./save/bankFL_{}_{}.png'.format(args.model, args.epochs))

    # testing
    train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs,
                              shuffle=True)
    # net_glob = torch.load('./models/net_nonp_bank.pkl')
    net_glob.eval()
    acc_train, loss_train = test_bank(net_glob, test_loader, args)
    acc_test, loss_test = test_bank(net_glob, train_loader, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
