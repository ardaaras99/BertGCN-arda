import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from random import seed

from model_scripts.layers import GraphConvolution


class GCN_type1(nn.Module):
    def __init__(self, A_s, v):
        super(GCN_type1, self).__init__()
        self.A_s = A_s
        self.v = v
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvolution(self.v.nfeat, self.v.n_hidden))
        self.gcn_layers.append(GraphConvolution(self.v.n_hidden, self.v.linear_h))
        self.linear = torch.nn.Linear(self.v.linear_h, self.v.nb_class)

    def forward(self, x):
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, self.A_s[i])
            if self.v.bn_activator[i] == "True":
                x = nn.BatchNorm1d(x.shape[1], affine=True).to(self.v.gpu)(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, self.v.dropout[i], training=self.training)

        x = self.linear(x)
        if self.v.bn_activator[-1] == "True":
            x = nn.BatchNorm1d(x.shape[1], affine=True).to(self.v.gpu)(x)
        return x


class GCN_type2(nn.Module):
    def __init__(self, A_s, cls_logit, v):
        super(GCN_type2, self).__init__()
        self.A_s = A_s
        self.cls_logit = cls_logit
        self.v = v
        self.gcn = GCN_type1(A_s=self.A_s, v=self.v)

    def forward(self, input_embeddings):
        gcn_logit = self.gcn(input_embeddings)
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        cls_pred = torch.nn.Softmax(dim=1)(self.cls_logit)
        pred = (gcn_pred + 1e-9) * self.v.m + cls_pred * (1 - self.v.m)
        pred = torch.log(pred)
        # pred is in form log softmax we will use nll loss
        return pred


# bert_model has bert and classifier component
class GCN_type3(nn.Module):
    def __init__(self, A_s, v):
        super(GCN_type3, self).__init__()
        self.A_s = A_s
        self.v = v
        # GCN Part
        self.gcn = GCN_type1(A_s=self.A_s, v=self.v)
        # BERT Part
        self.bert_clf = BertClassifier(self.v.bert_init, self.v.nb_class)
        self.feat_dim = list(self.bert_clf.bert_model.modules())[-2].out_features

    def forward(self, input_ids, attention_mask, gcn_input, idx):
        # idx -> current batch ids to update graph
        if self.training:
            cls_feats = self.bert_clf.bert_model(input_ids, attention_mask)[0][:, 0]
            # during training we update GCN inputs after BERT iteration
            gcn_input[idx] = cls_feats
        else:
            cls_feats = gcn_input[idx]

        cls_logit = self.bert_clf.classifier(cls_feats)
        cls_pred = nn.Softmax(dim=1)(cls_logit)

        gcn_logit = self.gcn(gcn_input)
        # burada softmax alıp idx hesaplamak la, idx alıp  softmax yapmak farklı şeyler
        # softmax için geri gpuya koymakta sorun yok
        gcn_pred = nn.Softmax(dim=1)(gcn_logit[idx])
        pred = (gcn_pred + 1e-10) * self.v.m + cls_pred * (1 - self.v.m)
        pred = torch.log(pred)
        return pred


class BertClassifier(torch.nn.Module):
    def __init__(self, pretrained_model="roberta_base", nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit
