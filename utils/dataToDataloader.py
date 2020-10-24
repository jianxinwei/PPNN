import torch
import numpy as np
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