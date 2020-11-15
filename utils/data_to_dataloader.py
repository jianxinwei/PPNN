import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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

def prngDataloader(attributes, labels, ids, batchsize=8, gene=None):
    ids = torch.LongTensor(list(ids))
    part_attributes = torch.index_select(attributes, 0, ids)
    part_labels = torch.index_select(labels, 0, ids)
    train_tensor = torch.utils.data.TensorDataset(part_attributes, part_labels)
    train_loader = DataLoader(dataset=train_tensor, batch_size=batchsize, shuffle=True, drop_last=True)
    return train_loader

def data_loading(args):
    if args.dataset != 'farm':
        wholedataset = pd.read_csv('../data/{}/new_{}_whole.csv'.format(args.dataset, args.dataset), sep=';')
        trainset, validset, testset = np.split(wholedataset, [int(args.train_ratio*len(wholedataset)), int((args.train_ratio + args.valid_ratio)*len(wholedataset))])

        train_attributes, train_labels = dfToTensor(trainset)
        train_attributes = train_attributes.to(args.device)
        train_labels = train_labels.to(args.device, dtype=torch.long)
        attrisize = list(train_attributes[0].size())[0]
        classes = args.num_classes

        valid_attributes, valid_labels = dfToTensor(validset)
        valid_attributes = valid_attributes.to(args.device)
        valid_labels = valid_labels.to(args.device, dtype=torch.long)

        test_attributes, test_labels = dfToTensor(testset)
        test_attributes = test_attributes.to(args.device)
        test_labels = test_labels.to(args.device, dtype=torch.long)
    else:
        farm_path = '../data/{}/farm-ads-vect'.format(args.dataset)
        farm_file = open(farm_path)
        lines = farm_file.readlines()
        farm_file.close()
        farm_vector = np.zeros([len(lines),55000])
        farm_label = np.zeros(len(lines))
        for l in range(len(lines)):
            line = lines[l].strip().split(' ')
            farm_label[l] = int(line[0])
            for i in range(1, len(line)):
                index = int(line[i].split(':')[0])
                farm_vector[l][index-1] = 1

        farm_label[farm_label < 0] = 0.0
        farm_label = farm_label.reshape(-1,1)

        farm_data = np.hstack((farm_vector,farm_label))
        wholedataset = pd.DataFrame(farm_data)

        trainset, validset, testset = np.split(wholedataset, [int(args.train_ratio*len(wholedataset)), int((args.train_ratio + args.valid_ratio)*len(wholedataset))])

        train_attributes, train_labels = dfToTensor(trainset)
        train_attributes = train_attributes.to(args.device)
        train_labels = train_labels.to(args.device, dtype=torch.long)
        attrisize = list(train_attributes[0].size())[0]
        classes = args.num_classes

        valid_attributes, valid_labels = dfToTensor(validset)
        valid_attributes = valid_attributes.to(args.device)
        valid_labels = valid_labels.to(args.device, dtype=torch.long)

        test_attributes, test_labels = dfToTensor(testset)
        test_attributes = test_attributes.to(args.device)
        test_labels = test_labels.to(args.device, dtype=torch.long)

    return train_attributes, train_labels, valid_attributes, valid_labels, test_attributes, test_labels