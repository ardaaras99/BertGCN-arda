from random import seed
import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT

import torch.nn as nn
import torch.nn.functional as F
from model.layers import *


class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class GCN_scratch(nn.Module):
    def __init__(self, nfeat, n_hidden, nclass, dropout):
        super(GCN_scratch, self).__init__()

        self.gc1 = GraphConvolution(nfeat, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, nclass)
        self.dropout = dropout

    def forward(self, x, NF, FN):
        x = F.relu(self.gc1(x, FN))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, NF)
        return x


class BertGCN_sparse(th.nn.Module):
    def __init__(self, nfeat, pretrained_model='roberta_base', nb_class=20, m=0.7, n_hidden=200, dropout=0.5):
        super(BertGCN_sparse, self).__init__()
        self.nfeat = nfeat
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        # new ones
        self.gcn = GCN_scratch(nfeat=self.nfeat,
                               n_hidden=n_hidden,
                               nclass=nb_class,
                               dropout=dropout)

    def forward(self, full_features, NF, FN, input_ids, attention_mask, doc_mask, idx):
        input_ids, attention_mask = input_ids[idx], attention_mask[idx]
        if self.training:
            features_batch = self.bert_model(
                input_ids, attention_mask)[0][:, 0]
            features_batch = features_batch.cuda()
            full_features[idx] = features_batch

            cls_logit = self.classifier(full_features)[idx]
            cls_pred = th.nn.Softmax(dim=1)(cls_logit)

            gcn_logit = self.gcn(full_features, NF, FN)[idx]
            gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
            #gcn_pred = gcn_out.max(1)[1].type_as(cls_pred)
            pred = (gcn_pred + 1e-10) * self.m + \
                (cls_pred+1e-10) * (1 - self.m)
            pred = th.log(pred + 1e-10)
        return pred


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        print("Feat_dim inside of BertGCN is: " + str(self.feat_dim))
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            # cls_feats -> train_size,768
            '''
                idx is same as batch size, it has indices of that current batch so we now which data
                instance we are dealing with
            '''
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'],
                             g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred


class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
            num_layers=gcn_layers-1,
            in_dim=self.feat_dim,
            num_hidden=n_hidden,
            num_classes=nb_class,
            heads=[heads] * (gcn_layers-1) + [1],
            activation=F.elu,
            feat_drop=dropout,
            attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
