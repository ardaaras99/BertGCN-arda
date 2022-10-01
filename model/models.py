from random import seed
import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN, GCN_scratch, GCN_scratch_2
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


class BertGCN_sparse_concat(th.nn.Module):
    def __init__(self, nfeat, pretrained_model='roberta_base', nb_class=20, m=0.7, n_hidden=200, dropout=0.5, A_s=None):
        super(BertGCN_sparse_concat, self).__init__()
        self.nfeat = nfeat
        self.m = m
        self.nb_class = nb_class
        self.n_hidden2 = 200
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim + self.n_hidden2, 300)
        # new ones
        self.A_s = A_s
        self.gcn = GCN_scratch_2(nfeat=self.nfeat,
                                 n_hidden1=n_hidden,
                                 n_hidden2=self.n_hidden2,
                                 dropout=dropout)

    def forward(self, g_input_ids, g_attention_mask, g_cls_feats, idx):

        input_ids, attention_mask = g_input_ids[idx], g_attention_mask[idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g_cls_feats[idx] = cls_feats
        else:
            cls_feats = g_cls_feats[idx]

        cls_out = self.classifier(cls_feats)

        x = g_cls_feats
        x = x.cuda()
        gcn_out = self.gcn(x, self.A_s)[idx]
        out_concat = th.concat((gcn_out, cls_feats), 1)
        out_logit = self.classifier(out_concat)

        return th.log(th.nn.Softmax(dim=1)(out_logit))


'''
    Sparse implementation of BertGCN with custom GCN Layer
'''


class BertGCN_sparse(th.nn.Module):
    def __init__(self, input_type, nfeat, pretrained_model='roberta_base', nb_class=20, m=0.7, n_hidden=[200], dropout=0.5, A_s=None):
        super(BertGCN_sparse, self).__init__()
        self.nfeat = nfeat
        self.m = m
        self.nb_class = nb_class
        self.input_type = input_type
        self.n_hidden = n_hidden
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        # new ones
        self.A_s = A_s
        self.gcn = GCN_scratch(A_s=self.A_s,
                               nfeat=self.nfeat,
                               n_hidden=self.n_hidden,
                               nclass=nb_class,
                               dropout=dropout)

    """
    g_cls_feats can be n_doc x 768 or n_word x n_word (I matrix)
    """

    def forward(self, g_input_ids, g_attention_mask, g_cls_feats, idx):

        input_ids, attention_mask = g_input_ids[idx], g_attention_mask[idx]
        if self.input_type == "document-matrix input":
            if self.training:
                cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
                g_cls_feats[idx] = cls_feats
            else:
                cls_feats = g_cls_feats[idx]
        else:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]

        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)

        x = g_cls_feats.cuda()
        gcn_logit = self.gcn(x)
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit[idx])

        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred


# class BertGCN(th.nn.Module):
#     def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
#         super(BertGCN, self).__init__()
#         self.m = m
#         self.nb_class = nb_class
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#         self.bert_model = AutoModel.from_pretrained(pretrained_model)
#         self.feat_dim = list(self.bert_model.modules())[-2].out_features
#         print("Feat_dim inside of BertGCN is: " + str(self.feat_dim))
#         self.classifier = th.nn.Linear(self.feat_dim, nb_class)
#         self.gcn = GCN(
#             in_feats=self.feat_dim,
#             n_hidden=n_hidden,
#             n_classes=nb_class,
#             n_layers=gcn_layers-1,
#             activation=F.elu,
#             dropout=dropout
#         )

#     def forward(self, g, idx):
#         input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
#         if self.training:
#             # cls_feats -> train_size,768
#             '''
#                 idx is same as batch size, it has indices of that current batch so we now which data
#                 instance we are dealing with
#             '''
#             cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
#             g.ndata['cls_feats'][idx] = cls_feats
#         else:
#             cls_feats = g.ndata['cls_feats'][idx]
#         cls_logit = self.classifier(cls_feats)
#         cls_pred = th.nn.Softmax(dim=1)(cls_logit)
#         gcn_logit = self.gcn(g.ndata['cls_feats'],
#                              g, g.edata['edge_weight'])[idx]
#         gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
#         pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
#         pred = th.log(pred)
#         return pred


# class BertGAT(th.nn.Module):

    # def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
    #     super(BertGAT, self).__init__()
    #     self.m = m
    #     self.nb_class = nb_class
    #     self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    #     self.bert_model = AutoModel.from_pretrained(pretrained_model)
    #     self.feat_dim = list(self.bert_model.modules())[-2].out_features
    #     self.classifier = th.nn.Linear(self.feat_dim, nb_class)
    #     self.gcn = GAT(
    #         num_layers=gcn_layers-1,
    #         in_dim=self.feat_dim,
    #         num_hidden=n_hidden,
    #         num_classes=nb_class,
    #         heads=[heads] * (gcn_layers-1) + [1],
    #         activation=F.elu,
    #         feat_drop=dropout,
    #         attn_drop=dropout,
    #     )

    # def forward(self, g, idx):
    #     input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
    #     if self.training:
    #         cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
    #         g.ndata['cls_feats'][idx] = cls_feats
    #     else:
    #         cls_feats = g.ndata['cls_feats'][idx]
    #     cls_logit = self.classifier(cls_feats)
    #     cls_pred = th.nn.Softmax(dim=1)(cls_logit)
    #     gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
    #     gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
    #     pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
    #     pred = th.log(pred)
    #     return pred
