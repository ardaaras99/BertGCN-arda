from random import seed
import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN_scratch, GCN_scratch_2


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

    def forward(self, g_input_ids, g_attention_mask, bert_output, idx):

        input_ids, attention_mask = g_input_ids[idx], g_attention_mask[idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            bert_output[idx] = cls_feats
        else:
            cls_feats = bert_output[idx]

        cls_out = self.classifier(cls_feats)

        x = bert_output
        x = x.cuda()
        gcn_out = self.gcn(x, self.A_s)[idx]
        out_concat = th.concat((gcn_out, cls_feats), 1)
        out_logit = self.classifier(out_concat)

        return th.log(th.nn.Softmax(dim=1)(out_logit))


'''
    Sparse implementation of BertGCN with custom GCN Layer
'''


class BertGCN_sparse(th.nn.Module):
    def __init__(self, input_type, nfeat, pretrained_model='roberta_base', nb_class=20, m=0.7, n_hidden=[200], dropout=0.5, A_s=None, train_bert_w_gcn="yes"):
        super(BertGCN_sparse, self).__init__()
        self.nfeat = nfeat
        self.nb_class = nb_class
        self.input_type = input_type
        self.n_hidden = n_hidden
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        # new ones
        self.A_s = A_s
        self.train_bert_w_gcn = train_bert_w_gcn
        # put constraint on learnable m it goes to -numbers
        #self.m = th.nn.Parameter(th.ones(1) * 0.5)
        self.m = m
        self.gcn = GCN_scratch(A_s=self.A_s,
                               nfeat=self.nfeat,
                               n_hidden=self.n_hidden,
                               nclass=nb_class,
                               dropout=dropout)

    def forward(self, g_input_ids, g_attention_mask, bert_output, gcn_input, idx):

        input_ids, attention_mask = g_input_ids[idx], g_attention_mask[idx]

        if self.input_type == "document-matrix input":

            if self.train_bert_w_gcn == "yes":
                if self.training:
                    cls_feats = self.bert_model(
                        input_ids, attention_mask)[0][:, 0]
                    gcn_input[idx] = cls_feats
                else:
                    cls_feats = bert_output[idx]
            else:
                cls_feats = bert_output[idx]

        elif self.input_type == "word-matrix input":
            # cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            cls_feats = bert_output[idx]

        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)

        gcn_logit = self.gcn(gcn_input)
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit[idx])

        pred = (gcn_pred) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
