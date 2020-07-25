import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Read data.
DATA_ROOT = "Datasets"
file_path = os.path.join(DATA_ROOT, 'aclImdb_v1.tar.gz')
if not os.path.exists(os.path.join(DATA_ROOT, 'aclImdb')):
    with tarfile.open(file_path, 'r') as f:
        f.extractall(DATA_ROOT)

train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')

## Pre process data.

# Selcet frequent words and create word dict.
# It includes a dict: Counter, stoi: dict, itos: list. 
vocab = d2l.get_vocab_imdb(train_data) 

# [[words index]..], [1, 0..]
features, labels = d2l.preprocess_imdb(train_data, vocab) 

batch_size = 64
# *: Put all return in a tuple.
train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab)) 
test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

## Define model.
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size, 
                               hidden_size=num_hiddens, 
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs.shape: (batch, words_num)
        # output: (words_num, batch, dim)
        embeddings = self.embedding(inputs.permute(1, 0))
        # output: (words_num, batch, 2 * num_hiddens)
        outputs, _ = self.encoder(embeddings) # output: (h, c)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

# Load pre-trained vectors.
glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(DATA_ROOT, "glove"))

net.embedding.weight.data.copy_(
    d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False 

lr, num_epochs = 0.01, 5
# Export none-graded embedding params.
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

## Train model.
model_name = 'IMDB_LSTM.pt'
model_path = os.path.join('Models', model_name)
if not os.path.exists(model_path):
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)
    torch.save(net, model_path)
else:
    net = torch.load(model_path)

## Predict.
reviews = ["the movie is amazing I like it",
           "I didn't enjoy it quite as much as the similar Princess Jellyfish (oh no, a Stylish!), but any who enjoy one will enjoy the other. Has some weird imagery, but largely average on an artistic sense. Some episodes get downright bad in technical quality."]
for r in reviews:
    print(r, '=>', d2l.predict_sentiment(net, vocab, r.split(' ')))





















