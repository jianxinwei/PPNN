#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        return x


class LargeMLP(nn.Module):
    def __init__(self, dim_in, args, dim_out):
        super(LargeMLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, args.farm_hidden_1)
        self.dropout_0 = nn.Dropout()
        self.relu_0 = nn.ReLU()
        self.layer_hidden_1 = nn.Linear(args.farm_hidden_1, args.farm_hidden_2)
        self.dropout_1 = nn.Dropout()
        self.relu_1 = nn.ReLU()
        self.layer_hidden_2 = nn.Linear(args.farm_hidden_2, args.farm_hidden_3)
        self.dropout_2 = nn.Dropout()
        self.relu_2 = nn.ReLU()
        self.layer_hidden_3 = nn.Linear(args.farm_hidden_3, args.farm_hidden_4)
        self.dropout_3 = nn.Dropout()
        self.relu_3 = nn.ReLU()
        self.layer_hidden_4 = nn.Linear(args.farm_hidden_4, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout_0(x)
        x = self.relu_0(x)
        x = self.layer_hidden_1(x)
        x = self.dropout_1(x)
        x = self.relu_1(x)
        x = self.layer_hidden_2(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)
        x = self.layer_hidden_3(x)
        x = self.dropout_3(x)
        x = self.relu_3(x)
        x = self.layer_hidden_4(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x