from utils import partition
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, autograd
import random
from sklearn import metrics


# dataframe to tensor
def dfToTensor(dataset):
    attributes = torch.Tensor(np.array(dataset.iloc[:, :-1].values.astype(np.float32)))
    labels = torch.Tensor(np.array(dataset.iloc[:, -1]))
    # print(labels)
    return attributes, labels


# split the dataset and put each of them in Dataloader
def clientDataloader(attributes, labels, ids, batchsize=8):
    ids = torch.LongTensor(list(ids))
    part_attributes = torch.index_select(attributes, 0, ids)
    part_labels = torch.index_select(labels, 0, ids)
    train_tensor = torch.utils.data.TensorDataset(part_attributes, part_labels)
    train_loader = DataLoader(dataset=train_tensor, batch_size=batchsize, shuffle=True)
    return train_loader


class LocalUpdate(object):
    def __init__(self, args, train_loader=None):
        self.args = args
        self.ldr_train = train_loader
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (attributes, labels) in enumerate(self.ldr_train):
                attributes, labels = attributes.to(self.args.device), labels.to(device=self.args.device, dtype=torch.long)
                net.zero_grad()
                log_probs = net(attributes)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(attributes), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
