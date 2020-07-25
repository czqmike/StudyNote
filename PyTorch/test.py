import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab

import d2lzh_pytorch as d2l

cache_dir = "Datasets/glove"
glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir) # 与上面等价

print(glove)
print("len(vectors) =", len(glove.vectors))

def knn(W, x, k):
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x.view((-1,))) / (
        (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.vectors,
                    embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))

get_similar_tokens('women', 3, glove)
