## Only use `Tensor` and `autograd` to create a linear regression training.
import torch
import numpy as np
from d2lzh_pytorch import *

## Generate data set.
## y = wX + b + c
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
## y = features colomn 0 * w[0] +  colomn 1 * w[1] + b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b 
labels += torch.tensor(
    np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

# print(features[0:10], labels[0:10])
# show_scatter(features[:, 1], labels)

## Initial model parameters.
w = torch.tensor(
    np.random.normal(0, 0.01, (num_inputs, 1)), 
    dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

## Train model.
lr = 0.03 # learning rate
num_epoches = # SGD iteration cycle
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epoches):
    for X, y in data_iter(batch_size, features, labels):
        ## l is not a scalar, so sum() it that can be backward()
        l = loss(net(X, w, b), y).sum() 
        l.backward()
        sgd([w, b], lr, batch_size)

        ## Don't forget clear grad.
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print("epoch %d, loss %f" % (epoch + 1, train_l.mean().item() ))

print("w: ", w, "\nture_w: ", true_w)
print("b: ", b, "\nture_b: ", true_b)