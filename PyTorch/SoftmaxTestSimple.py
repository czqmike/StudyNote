import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch as d2l
from collections import OrderedDict

device = torch.device(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
# device = torch.device(torch.device('cpu'))

## Load data.
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

## Define model.
net = nn.Sequential(
    # FlattenLayer()
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', d2l.FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
).to(device)

## Init paras.
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

## Define loss function.
loss = nn.CrossEntropyLoss().to(device)

## Define optimization function.
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

## Train model.
num_epoches = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoches,
              batch_size, None, None, optimizer)

X, y = iter(test_iter).next()

net = net.cpu()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
title = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[:9], title[:9], 10)