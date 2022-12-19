"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""

import torch as th
from utils_train_v2 import *
from utils_v2 import *
import torch.nn as nn
from layers import GraphConvolution
import torch.nn.functional as F


from random import seed
import torch as th
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from gcn_models import *

import torch.nn.functional as F
from layers import *


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


class GCN_type1(nn.Module):
    def __init__(self, A_s, nfeat, v, gpu, nclass):
        super(GCN_type1, self).__init__()
        self.A_s = A_s
        self.v = v
        self.gcn_layers = nn.ModuleList()
        self.gpu = gpu
        current_dim = nfeat
        for hdim in self.v.n_hidden:
            self.gcn_layers.append(GraphConvolution(current_dim, hdim))
            current_dim = hdim
        self.gcn_layers.append(GraphConvolution(current_dim, nclass))

    def forward(self, x):
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, self.A_s[i])
            if i != len(self.gcn_layers) - 1:  # removing last BN before softmax
                m = nn.BatchNorm1d(x.shape[1], affine=True).to(self.gpu)
                x = m(x)
            x = F.leaky_relu(x)
        x = F.dropout(x, self.v.dropout, training=self.training)
        # no log softmax here, it will be done in combined model
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
        self.gcn = GCN_type1(A_s=self.A_s,
                             nfeat=self.nfeat,
                             v=self.v,
                             gpu=self.gpu,
                             nclass=self.n_class)
        # self.gcn.load_state_dict(th.load('gcn_models/{}_type1_weights_{}.pt'.format(
        #     v.dataset, v.gcn_path)))

    def forward(self, input_embeddings):
        self.gcn.to(self.gpu)
        gcn_logit = self.gcn(input_embeddings)

        cls_pred = th.nn.Softmax(dim=1)(self.cls_logit)
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred) * self.v.m + cls_pred * (1 - self.v.m)
        pred = th.log(pred)
        # pred is in form log softmax we will use nll loss
        return pred


class GCN_Trainer:
    def __init__(self, model, optimizer, scheduler, label,
                 input_embeddings, nb_train, nb_val, nb_test,
                 v, gpu, criterion,
                 patience=200, print_gap=10,
                 model_path=''):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label = label
        self.input_embeddings = input_embeddings
        self.nb_train = nb_train
        self.nb_val = nb_val
        self.nb_test = nb_test
        self.v = v
        self.gpu = gpu
        self.criterion = criterion

        self.patience = patience
        self.print_gap = print_gap
        self.model_path = model_path

    def train_model(self):
        self.model.to(self.gpu)
        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(self.input_embeddings)[:self.nb_train]
        y_true = self.label['train'].type(th.long).to(self.gpu)

        loss = self.criterion(y_pred, y_true)
        w_f1, macro, micro = get_metrics(y_pred, y_true)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        train_loss = loss.item()

        return train_loss, w_f1, macro, micro

    def eval_model(self, phase):
        with th.no_grad():
            self.model.to(self.gpu)
            self.model.eval()

            y_pred = self.model(self.input_embeddings)

            if phase == 'val':
                y_pred = y_pred[self.nb_train:self.nb_train + self.nb_val]
            else:
                y_pred = y_pred[-self.nb_test:]

            y_true = self.label[phase].type(th.long).to(self.gpu)
            loss = self.criterion(y_pred, y_true)
            test_loss = loss.item()
            w_f1, macro, micro = get_metrics(y_pred, y_true)
        return test_loss, w_f1, macro, micro

    def print_results(self, train_w_f1, train_macro, train_micro,
                      val_w_f1, val_macro, val_micro,
                      train_loss, val_loss,
                      epoch):

        epoch_len = len(str(self.v.nb_epochs))
        if epoch % self.print_gap == 0:
            print_msg = (f'[{epoch:>{epoch_len}}/{self.v.nb_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.3f} ' +
                         f'valid_loss: {val_loss:.3f} ' +
                         f'train_w_f1: {100*train_w_f1:.3f} ' +
                         f'val_w_f1: {100*val_w_f1:.3f} ')

            print(print_msg)

    def train_val_loop(self):
        self.model.to(self.gpu)
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=False, path=self.model_path)

        for epoch in range(1, self.v.nb_epochs+1):

            train_loss, train_w_f1, train_macro, train_micro = self.train_model()
            val_loss, val_w_f1, val_macro, val_micro = self.eval_model(
                phase='val')
            self.print_results(train_w_f1, train_macro, train_micro,
                               val_w_f1, val_macro, val_micro,
                               train_loss, val_loss,
                               epoch)

            early_stopping(val_micro, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.model.load_state_dict(th.load(self.model_path))

        return self.model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + 0.0000001:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(
                f'Validation acc increased ({100*self.val_acc_min:.3f} --> {100*val_acc:.3f}).  Saving model ...')
        th.save(model.state_dict(), self.path)
        self.val_acc_min = val_acc
