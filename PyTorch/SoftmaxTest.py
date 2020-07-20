import torch
import torchvision
import numpy as np
import d2lzh_pytorch as d2l


## Load data.
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

## Init model parameters.
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), 
                 dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(True)
b.requires_grad_(True)

## Define softmax operation.
def softmax(X):
    X_exp = X.exp() # transform to non-negative number
    partition = X_exp.sum(dim=1, keepdim=True) # sumed by rows
    return X_exp / partition # broadcast used here

## Define model.
def net(X):
    ## -1 stands for single dimension
    return softmax(torch.mm( X.view((-1, num_inputs)), W )) 

## Define loss function.
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

num_epoches, lr = 5, 0.1

## Train model.
d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epoches, batch_size, [W, b], lr)

## Eval model.
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
title = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[:9], title[:9], 10)