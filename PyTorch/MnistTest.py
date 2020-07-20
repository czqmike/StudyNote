import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import d2lzh_pytorch as d2l

## Download data set.
mnist_train = torchvision.datasets.FashionMNIST(
    root='~/StudyNote/PyTorch/Datasets/FashionMNIST', train=True, 
    download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(
    root='~/StudyNote/PyTorch/Datasets/FashionMNIST', train=False, 
    download=True, transform=transforms.ToTensor())

## Peek dataset.
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0]) # feature
    y.append(mnist_train[i][1]) # label in number
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y), 10)

batch_size = 256
num_workers = 4 # number of multi processes
import torch.utils.data as tdata
train_iter = tdata.DataLoader(mnist_train, batch_size=batch_size, 
             shuffle=True, num_workers=num_workers)
test_iter = tdata.DataLoader(mnist_test, batch_size=batch_size, 
             shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))