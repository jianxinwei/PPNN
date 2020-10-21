#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F


def test_bank(net_g, test_loader, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    l = len(test_loader)
    for idx, (data, target) in enumerate(test_loader):
        # if args.gpu != -1:
        #     data, target = data.cuda(), target.cuda()
        target = target.to(dtype=torch.long)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.00 * correct / len(test_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy, test_loss