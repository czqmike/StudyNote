from matplotlib import pyplot as plt
import torch
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as transforms
import random
import time
import d2lzh_pytorch as d2l

def show_scatter(x, y, timeParsed=5):
    plt.scatter(x, y, 1)
    plt.show(block=False)
    plt.pause(timeParsed)
    plt.close()

## Return several (batch_size) features and labels every time.
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) ## Randomly read data.
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

## Linear regression expression.
def linreg(X, w, b):
    return torch.mm(X, w) + b

## Loss function of linear regression. (a - b) ^ 2 / 2
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()) ) ** 2 / 2

## Stochastic Gradient Descent in small batches.
def sgd(params, lr, batch_size):
    for param in params:
        ## Use [param.data] to change param
        if param.grad is not None:
            param.data -= lr * param.grad / batch_size 

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels, timeParsed=5):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show(block=False)
    plt.pause(timeParsed)
    plt.close()

## return: train_iter, test_iter
## Use: for feature, label in train_iter..
def load_data_fashion_mnist(batch_size=256, num_workers=4):
    mnist_train = torchvision.datasets.FashionMNIST(
        root='~/StudyNote/PyTorch/Datasets/FashionMNIST', train=True, 
        download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(
        root='~/StudyNote/PyTorch/Datasets/FashionMNIST', train=False, 
        download=True, transform=transforms.ToTensor())
    train_iter = tdata.DataLoader(mnist_train, batch_size=batch_size, 
                shuffle=True, num_workers=num_workers)
    test_iter = tdata.DataLoader(mnist_test, batch_size=batch_size, 
                shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device(torch.device('cpu'))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.to(device=device)
        y = y.to(device=device)
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        startTime = time.time() 
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device=device)
            y = y.to(device=device)
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    if param.grad is not None:
                        param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        duration = time.time() - startTime
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.3f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, duration))

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5), timeParsed=10):
    # d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    plt.show(block=False)
    plt.pause(timeParsed)
    plt.close()