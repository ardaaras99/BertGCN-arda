"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
# from dgl.nn import GraphConv
from .graphconv_edge_weight import GraphConvEdgeWeight as GraphConv
from .layers import GraphConvolution
import torch.nn.functional as F


class GCN_scratch(nn.Module):
    def __init__(self, A_s, nfeat, n_hidden, nclass, dropout):
        super(GCN_scratch, self).__init__()
        self.A_s = A_s
        self.layer_no = len(self.A_s)
        self.n_hidden = n_hidden
        self.gcn_layers = nn.ModuleList()
        self.dropout = dropout

        current_dim = nfeat
        for hdim in n_hidden:
            self.gcn_layers.append(GraphConvolution(current_dim, hdim))
            current_dim = hdim
        self.gcn_layers.append(GraphConvolution(current_dim, nclass))

    def forward(self, x):
        for i, layer in enumerate(self.gcn_layers):
            x = F.leaky_relu(layer(x, self.A_s[i]))
            x = F.dropout(x, self.dropout, training=self.training)
        # no log softmax here, it will be done in combined model
        return x


class GCN_scratch_2(nn.Module):
    def __init__(self, nfeat, n_hidden1, n_hidden2=100, dropout=0.5):
        super(GCN_scratch_2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, n_hidden1)
        self.gc2 = GraphConvolution(n_hidden1, n_hidden2)
        self.dropout = dropout

    def forward(self, x, A1, A2):
        x = F.leaky_relu(self.gc1(x, A1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, A2)
        # no log softmax here, it will be done in combined model
        return x


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden,
                           activation=activation, norm=normalization))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,
                               activation=activation, norm=normalization))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, norm=normalization))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, g, edge_weight):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weight)
        return h
