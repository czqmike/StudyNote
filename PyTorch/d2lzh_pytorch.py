from matplotlib import pyplot as plt

def show_scatter(x, y, timeParsed=5):
    plt.scatter(x, y, 1)
    plt.show(block=False)
    plt.pause(timeParsed)
    plt.close()

import torch
import random
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
        param.data -= lr * param.grad / batch_size 