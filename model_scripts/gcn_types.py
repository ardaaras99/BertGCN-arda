import torch.nn as nn
import torch as th
import torch.nn.functional as F

from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from random import seed

from model_scripts.layers import GraphConvolution


class GCN_type1(nn.Module):
    def __init__(self, A_s, nfeat, v, gpu, nclass):
        super(GCN_type1, self).__init__()
        self.A_s = A_s
        self.v = v
        self.gcn_layers = nn.ModuleList()
        self.gpu = gpu
        self.current_dim = nfeat
        for hdim in self.v.n_hidden:
            self.gcn_layers.append(GraphConvolution(self.current_dim, hdim))
            self.current_dim = hdim
        self.gcn_layers.append(GraphConvolution(self.current_dim, self.v.linear_h))
        self.linear = th.nn.Linear(self.v.linear_h, nclass)

    def forward(self, x):
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, self.A_s[i])
            if self.v.bn_activator[i] == "True":
                x = nn.BatchNorm1d(x.shape[1], affine=True).to(self.gpu)(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, self.v.dropout[i], training=self.training)

        x = self.linear(x)
        if self.v.bn_activator[-1] == "True":
            x = nn.BatchNorm1d(x.shape[1], affine=True).to(self.gpu)(x)
        return x


class GCN_type2(nn.Module):
    def __init__(self, A_s, nfeat, v, gpu, cls_logit, n_class):
        super(GCN_type2, self).__init__()
        self.A_s = A_s
        self.v = v
        self.nfeat = nfeat
        self.gpu = gpu
        self.cls_logit = cls_logit
        self.n_class = n_class
        self.gcn = GCN_type1(
            A_s=self.A_s, nfeat=self.nfeat, v=self.v, gpu=self.gpu, nclass=self.n_class
        )

        # tek başına type1 trainletip ordan başlatalım dedik ama çok da güzel olmadı
        # self.gcn.load_state_dict(th.load('gcn_models/{}_type1_weights_{}.pt'.format(
        #     v.dataset, v.gcn_path)))

    def forward(self, input_embeddings):
        self.gcn.to(self.gpu)
        gcn_logit = self.gcn(input_embeddings)
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)

        cls_pred = th.nn.Softmax(dim=1)(self.cls_logit)
        pred = (gcn_pred) * self.v.m + cls_pred * (1 - self.v.m)
        pred = th.log(pred)
        # pred is in form log softmax we will use nll loss
        return pred


class GCN_type3(nn.Module):
    def __init__(self, A_s, nfeat, v, v_bert, gpu, n_class):
        super(GCN_type3, self).__init__()
        self.A_s = A_s
        self.v = v
        self.nfeat = nfeat
        self.gpu = gpu
        self.n_class = n_class
        # GCN Part
        self.gcn = GCN_type1(
            A_s=self.A_s, nfeat=self.nfeat, v=self.v, gpu=self.gpu, nclass=self.n_class
        )
        # BERT Part
        self.v_bert = v_bert
        self.bert = BertClassifier(self.v_bert.bert_init, self.n_class)
        self.feat_dim = list(self.bert.bert_model.modules())[-2].out_features

    def forward(self, input_ids, attention_mask, gcn_input, idx):
        # idx -> current batch ids to update graph
        if self.training:
            cls_feats = self.bert.bert_model(input_ids, attention_mask)[0][:, 0]
            # during training we update GCN inputs after BERT iteration
            gcn_input[idx] = cls_feats
        else:
            cls_feats = gcn_input[idx]

        cls_logit = self.bert.classifier(cls_feats)
        cls_pred = nn.Softmax(dim=1)(cls_logit)

        gcn_logit = self.gcn(gcn_input)
        # burada softmax alıp idx hesaplamak la, idx alıp  softmax yapmak farklı şeyler
        gcn_pred = nn.Softmax(dim=1)(gcn_logit[idx])

        pred = (gcn_pred + 1e-10) * self.v.m + cls_pred * (1 - self.v.m)
        pred = th.log(pred)
        return pred


class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model="roberta_base", nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit
