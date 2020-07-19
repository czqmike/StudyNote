import torch
import torch.utils.data as tdata
import torch.nn as nn
import torch.optim as optim
import numpy as np

## Generate data set.
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

## Read data.
batch_size = 10

dataset = tdata.TensorDataset(features, labels) # Combine features and labels.
data_iter = tdata.DataLoader(dataset, batch_size, shuffle=True)

## Define model.
# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(n_feature, 1)
    
#     def forward(self, x):
#         y = self.linear(x)
#         return y

# net = LinearNet(num_inputs)

## Another way to define model.
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

## Initial model parameters.
nn.init.normal_(net[0].weight, mean=0, std=0.01)
nn.init.constant_(net[0].bias, val=0)

## Define loss function.
loss = nn.MSELoss()

## Define optimization algorithm (SGD)
optimizer = optim.SGD(net.parameters(), lr=0.03)

## Train model.
num_epoches = 3
for epoch in range(1, num_epoches + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # Clear grad <=> net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

print("w: ", net[0].weight, "\nture_w: ", true_w)
print("b: ", net[0].bias, "\nture_b: ", true_b)