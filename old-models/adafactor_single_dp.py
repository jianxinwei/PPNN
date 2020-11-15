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
from utils.optim import *
from utils.utils import *

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

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
    # w_glob = net_glob.state_dict()

    # training
    loss_train = []
    net_best = None
    best_loss = None

    optimizer = Adafactor(net_glob.parameters())
    train_loader = DataLoader(dataset=TensorDataset(train_attributes, train_labels), batch_size=args.bs, shuffle=True)
    loss_func = nn.CrossEntropyLoss()

    args.secure_rng = True
    privacy_engine = PrivacyEngine(net_glob, batch_size=args.bs, sample_size=len(train_loader),
                                   alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                                   noise_multiplier=0.3, max_grad_norm=1.2, secure_rng=args.secure_rng)
    privacy_engine.attach(optimizer)

    loss_valid = []
    best_valid_loss = np.finfo(float).max
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

            '''
            if batch_idx % args.pb_interval == 0:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
                print(
                    f"\tTrain Epoch: {iter} \t"
                    f"Loss: {loss_avg:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta})"
                )
            '''
            net_glob.eval()
            acc_valid, tmp_loss_valid = test_bank(net_glob, valid_loader, args)
            print('Round{:3d}, Validation loss {:.3f}'.format(iter, tmp_loss_valid))
            loss_valid.append(tmp_loss_valid)
            if tmp_loss_valid < best_valid_loss:
                best_valid_loss = tmp_loss_valid
                torch.save(net_glob, './save/adafactor_single_dp_best_{}.pkl'.format(args.dataset))
                print('SAVE BEST MODEL AT EPOCH {}'.format(iter))
            net_glob.train()

    torch.save(net_glob, './save/adafactor_single_dp_final_{}.pkl'.format(args.dataset))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train, 'ro-', label='train_loss')
    plt.plot(range(len(loss_valid)), loss_valid, 'b^-', label='valid_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig('./save/{}_adafactor_single_dp_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    net_glob = torch.load('./save/adafactor_single_dp_best_{}.pkl'.format(args.dataset))
    net_glob.eval()
    if args.gpu != -1:
        test_loader = DataLoader(dataset=TensorDataset(test_attributes, test_labels), batch_size=args.bs, shuffle=True)
    acc_train, loss_train = test_bank(net_glob, train_loader, args)
    acc_valid, loss_valid = test_bank(net_glob, valid_loader, args)
    acc_test, loss_test = test_bank(net_glob, test_loader, args)
    print("Training accuracy: {:.2f}".format(acc_train), "Training loss: {:.2f}".format(loss_train))
    print("Validating accuracy: {:.2f}".format(acc_valid), "Validating loss: {:.2f}".format(loss_valid))
    print("Testing accuracy: {:.2f}".format(acc_test), "Testing loss: {:.2f}".format(loss_test))