# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

n_samples = 100
n_features = 20
A_ = sp.random(n_samples, n_samples, dtype=np.float32)
X_ = sp.random(n_samples, n_samples, dtype=np.float32)
# ↑here


def to_torch_sparse_tensor(M):
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    T = torch.sparse.FloatTensor(indices, values, shape)
    return T


X = to_torch_sparse_tensor(X_)
A = to_torch_sparse_tensor(A_)


class GraphConv(nn.Module):
    def __init__(self, size_in, size_out):
        super(GraphConv, self).__init__()
        print(size_in, size_out)
        self.W = nn.parameter.Parameter(torch.Tensor(size_in, size_out))
        self.b = nn.parameter.Parameter(torch.Tensor(size_out))
        # Initialize weights
        variance = 2 / (size_in + size_out)
        self.W.data.normal_(0.0, variance)
        self.b.data.normal_(0.0, variance)

    def forward(self, X, A):
        return torch.mm(torch.mm(A, X), self.W) + self.b


gcn = GraphConv(X.size()[1], 128)
gcn(X, A)
