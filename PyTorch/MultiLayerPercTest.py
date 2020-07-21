## Create MLP (Multi Layer Perceptron) from 0.
import torch
import numpy as np
import d2lzh_pytorch as d2l

## Load data.
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), 
                  dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), 
                  dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2]

## Define activative function.
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

## Define model.
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1) # H = relu(XW1 + b1)
    return torch.matmul(H, W2) + b2 # O = HW2 + b2

## Define loss function.
loss = torch.nn.CrossEntropyLoss()

## Train model.
num_epoches, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, params, lr)