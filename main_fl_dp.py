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
    # trainset = pd.read_csv('./data/{}/new_{}_full.csv'.format(args.dataset, args.dataset), sep=';')
    # testset = pd.read_csv('./data/{}/new_{}.csv'.format(args.dataset, args.dataset), sep=';')
    wholedataset = pd.read_csv('./data/{}/new_{}_whole.csv'.format(args.dataset, args.dataset), sep=';')
    trainset, validset, testset = np.split(wholedataset, [int(args.train_ratio*len(wholedataset)), int((args.train_ratio + args.valid_ratio)*len(wholedataset))])

    train_attributes, train_labels = dfToTensor(trainset)
    train_attributes = train_attributes.to(args.device)
    train_labels = train_labels.to(args.device, dtype=torch.long)
    attrisize = list(train_attributes[0].size())[0]
    classes = args.num_classes

    valid_attributes, valid_labels = dfToTensor(validset)
    valid_attributes = valid_attributes.to(args.device)
    valid_labels = valid_labels.to(args.device, dtype=torch.long)
    valid_loader = DataLoader(dataset=TensorDataset(valid_attributes, valid_labels), batch_size=args.bs, shuffle=True)

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

    loss_valid = []
    best_valid_loss = np.finfo(float).max
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
            net_glob.eval()
            acc_valid, tmp_loss_valid = test_bank(net_glob, valid_loader, args)
            print('Round{:3d}, Validation loss {:.3f}'.format(iter, tmp_loss_valid))
            loss_valid.append(tmp_loss_valid)
            if tmp_loss_valid < best_valid_loss:
                best_valid_loss = tmp_loss_valid
                torch.save(net_glob, './save/fl_dp_best_{}.pkl'.format(args.dataset))
                print('SAVE BEST MODEL AT EPOCH {}'.format(iter))
            net_glob.train()

    torch.save(net_glob, './save/fl_dp_final_{}.pkl'.format(args.dataset))


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train, 'ro-', label='train_loss')
    plt.plot(range(len(loss_valid)), loss_valid, 'b^-', label='valid_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig('./save/{}_fl_dp_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
    net_glob = torch.load('./save/fl_dp_best_{}.pkl'.format(args.dataset))
    net_glob.eval()
    acc_train, loss_train = test_bank(net_glob, train_loader, args)
    acc_valid, loss_valid = test_bank(net_glob, valid_loader, args)
    acc_test, loss_test = test_bank(net_glob, test_loader, args)
    print("Training accuracy: {:.2f}".format(acc_train), "Training loss: {:.2f}".format(loss_train))
    print("Validating accuracy: {:.2f}".format(acc_valid), "Validating loss: {:.2f}".format(loss_valid))
    print("Testing accuracy: {:.2f}".format(acc_test), "Testing loss: {:.2f}".format(loss_test))