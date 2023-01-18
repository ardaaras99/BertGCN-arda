"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""

import torch as th
from utils_v2 import *
import torch.nn as nn
from layers import GraphConvolution
import torch.nn.functional as F
import time

from random import seed
import torch as th
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from gcn_models import *

from torch.optim import lr_scheduler
import torch.nn.functional as F
from layers import *

set_seed()


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


class BertTrainer:
    def __init__(
        self,
        v,
        v_bert,
        gpu,
        cpu,
        model,
        optimizer,
        scheduler,
        criterion,
        loader,
        dataset_sizes,
        label,
        model_type="fine_tune",
    ):

        self.v = v
        self.v_bert = v_bert
        self.gpu = gpu
        self.cpu = cpu
        self.model_type = model_type
        self.model = model

        # BURASI KRİTİK BAYA DOĞRU ÇALIŞTIĞINA EMİN OL
        if model_type == "fine_tune":
            self.model_weights_path = "bert-finetune_models/{}_weights.pt".format(
                self.v_bert.dataset
            )
        else:
            self.model_weights_path = "gcn_models/{}_type{}_weights_{}.pt".format(
                v.dataset, str(3), v.gcn_path
            )
            self.model.bert.load_state_dict(
                torch.load("bert-finetune_models/{}_weights.pt".format(v.dataset))
            )

        # self e save edilmesi gerekenler:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.loader = loader
        self.dataset_sizes = dataset_sizes
        self.label = label

    def get_gcn_input(self):
        with th.no_grad():
            self.model.to(self.gpu)
            self.model.eval()
            phases = ["train", "val", "test"]
            cls_list = []
            for phase in phases:
                for iis, att_mask, _, _ in self.loader[phase]:
                    cls_feats = self.model.bert.bert_model(
                        iis.to(self.gpu), att_mask.to(self.gpu)
                    )[0][:, 0]

                    cls_list.append(cls_feats.to(self.cpu))
            input_embeddings = th.cat(cls_list, axis=0)  # type: ignore
        return input_embeddings

    def loop(self, phase="train"):
        loss_s, w_f1, macro, micro, acc = 0, 0, 0, 0, 0
        self.model.to(self.gpu)
        self.model.train() if phase == "train" else self.model.eval()

        if self.model_type != "fine_tune":
            gcn_input = self.get_gcn_input().to(self.gpu)
        else:
            gcn_input = 0

        for iis, att_mask, lbls, idx in self.loader[phase]:

            iis, att_mask, lbls, idx = (
                iis.to(self.gpu),
                att_mask.to(self.gpu),
                lbls.to(self.gpu),
                idx.to(self.gpu),
            )

            if phase == "train":
                self.optimizer.zero_grad()

            y_true = lbls.type(th.long)
            y_pred = self.model(iis, att_mask, gcn_input, idx)  # type: ignore
            loss = F.cross_entropy(y_pred, y_true)
            # t for temp
            w_f1_t, macro_t, micro_t, acc_t = get_metrics(y_pred, y_true)

            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            norm_c = y_pred.shape[0] / self.dataset_sizes[phase][0]
            loss_s += loss.item() * norm_c
            w_f1 += w_f1_t * norm_c
            macro += macro_t * norm_c
            micro += micro_t * norm_c
            acc += acc_t * norm_c
        return loss_s, w_f1, macro, micro, acc

    def print_results(self, train_w_f1, val_w_f1, train_loss, val_loss, epoch):

        epoch_len = len(str(self.v_bert.nb_epochs))
        if epoch % self.v_bert.print_gap == 0:
            print_msg = (
                f"[{epoch:>{epoch_len}}/{self.v_bert.nb_epochs:>{epoch_len}}] "
                + f"train_loss: {train_loss:.3f} "
                + f"valid_loss: {val_loss:.3f} "
                + f"train_w_f1: {100*train_w_f1:.3f} "
                + f"val_w_f1: {100*val_w_f1:.3f} "
            )

            print(print_msg)

    def train_val_loop(self):
        early_stopping = EarlyStopping(
            patience=self.v_bert.patience, verbose=True, path=self.model_weights_path
        )

        for epoch in range(1, self.v_bert.nb_epochs + 1):
            start = time.time()
            train_loss, train_w_f1, _, _, _ = self.loop(phase="train")

            with th.no_grad():
                val_loss, val_w_f1, _, val_micro, _ = self.loop(phase="val")

            self.print_results(train_w_f1, val_w_f1, train_loss, val_loss, epoch)

            early_stopping(val_micro, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            end = time.time()
            if self.model_type != "fine_tune" and epoch == 1:
                print("GCN Type3 time per epoch: {:.3f}".format(end - start))
        self.model.load_state_dict(torch.load(self.model_weights_path))
        return self.model

    # This method specific to fine-tune bert structure
    def generate_embeddings_and_logits(self):
        with th.no_grad():
            self.model.to(self.gpu)
            self.model.eval()
            phases = ["train", "val", "test"]
            cls_list = []
            cls_logit_list = []
            for phase in phases:
                for iis, att_mask, _, _ in self.loader[phase]:
                    cls_feats = self.model.bert_model(
                        iis.to(self.gpu), att_mask.to(self.gpu)
                    )[0][
                        :, 0
                    ]  # original bert_model has classifier head we get rid of it
                    cls_logits = self.model(iis.to(self.gpu), att_mask.to(self.gpu))

                    cls_list.append(cls_feats.to(self.cpu))
                    cls_logit_list.append(cls_logits.to(self.cpu))
            input_embeddings = th.cat(cls_list, axis=0)  # type: ignore
            cls_logits = th.cat(cls_logit_list, axis=0)  # type: ignore

        torch.save(
            input_embeddings,
            "bert-finetune_models/{}_embeddings.pt".format(self.v_bert.dataset),
        )
        torch.save(
            cls_logits, "bert-finetune_models/{}_logits.pt".format(self.v_bert.dataset)
        )


class GCN_Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        label,
        input_embeddings,
        nb_train,
        nb_val,
        nb_test,
        v,
        gpu,
        criterion,
        model_path="",
    ):
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

        self.model_path = model_path

    def train_model(self):
        self.model.to(self.gpu)
        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(self.input_embeddings)[: self.nb_train]
        y_true = self.label["train"].type(th.long).to(self.gpu)

        loss = self.criterion(y_pred, y_true)
        w_f1, macro, micro, acc = get_metrics(y_pred, y_true)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        train_loss = loss.item()

        return train_loss, w_f1, macro, micro, acc

    def eval_model(self, phase):
        with th.no_grad():
            self.model.to(self.gpu)
            self.model.eval()

            y_pred = self.model(self.input_embeddings)

            if phase == "val":
                y_pred = y_pred[self.nb_train : self.nb_train + self.nb_val]
            else:
                y_pred = y_pred[-self.nb_test :]

            y_true = self.label[phase].type(th.long).to(self.gpu)
            loss = self.criterion(y_pred, y_true)
            test_loss = loss.item()
            w_f1, macro, micro, acc = get_metrics(y_pred, y_true)
        return test_loss, w_f1, macro, micro, acc

    def print_results(
        self,
        train_w_f1,
        train_macro,
        train_micro,
        val_w_f1,
        val_macro,
        val_micro,
        val_acc,
        train_loss,
        val_loss,
        epoch,
    ):

        epoch_len = len(str(self.v.nb_epochs))
        if epoch % self.v.print_gap == 0:
            print_msg = (
                f"[{epoch:>{epoch_len}}/{self.v.nb_epochs:>{epoch_len}}] "
                + f"train_loss: {train_loss:.3f} "
                + f"valid_loss: {val_loss:.3f} "
                + f"train_w_f1: {100*train_w_f1:.3f} "
                + f"val_w_f1: {100*val_w_f1:.3f} "
                + f"val_acc: {100*val_acc:.3f} "
            )

            print(print_msg)

    def train_val_loop(self):
        self.model.to(self.gpu)
        early_stopping = EarlyStopping(
            patience=self.v.patience, verbose=False, path=self.model_path
        )
        avg_time = []
        for epoch in range(1, self.v.nb_epochs + 1):
            start = time.time()
            self.v.current_epoch = epoch
            (
                train_loss,
                train_w_f1,
                train_macro,
                train_micro,
                train_acc,
            ) = self.train_model()
            val_loss, val_w_f1, val_macro, val_micro, val_acc = self.eval_model(
                phase="val"
            )
            self.print_results(
                train_w_f1,
                train_macro,
                train_micro,
                val_w_f1,
                val_macro,
                val_micro,
                val_acc,
                train_loss,
                val_loss,
                epoch,
            )

            early_stopping(val_micro, self.model)
            end = time.time()
            avg_time.append(end - start)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        avg_time = np.mean(avg_time)

        self.model.load_state_dict(th.load(self.model_path))

        return self.model, avg_time


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


class GCN_type1(nn.Module):
    def __init__(self, A_s, nfeat, v, gpu, nclass):
        super(GCN_type1, self).__init__()
        self.A_s = A_s
        self.v = v
        self.gcn_layers = nn.ModuleList()
        self.gpu = gpu
        self.current_dim = nfeat
        # hidden dimde yer alan eleman kadar GCN layeri ekle
        for hdim in self.v.n_hidden:
            self.gcn_layers.append(GraphConvolution(self.current_dim, hdim))
            self.current_dim = hdim
        # en son classification için gcn layeri ekle
        # belki bundan önce bir linear layer ekleyebilirsin?
        self.gcn_layers.append(GraphConvolution(self.current_dim, self.v.linear_h))
        self.linear = th.nn.Linear(self.v.linear_h, nclass)

    def forward(self, x):
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, self.A_s[i])
            # removing last BN before softmax
            if self.v.bn_activator[i] == "True":
                x = nn.BatchNorm1d(x.shape[1], affine=True).to(self.gpu)(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, self.v.dropout[i], training=self.training)

        x = self.linear(x)
        if self.v.bn_activator[-1] == "True":
            x = nn.BatchNorm1d(x.shape[1], affine=True).to(self.gpu)(x)

        # x = F.leaky_relu(x)
        # x = F.dropout(x, self.v.dropout[-1], training=self.training)
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


class Type_Trainer:
    def __init__(self, v, v_bert, all_paths, gpu, cpu, gcn_type, seed_no=42):
        self.v = v
        self.v_bert = v_bert
        self.all_paths = all_paths
        self.gpu = gpu
        self.cpu = cpu
        self.gcn_type = gcn_type
        (
            self.docs,
            self.y,
            train_ids,
            test_ids,
            self.NF,
            self.FN,
            self.NN,
            self.FF,
        ) = load_corpus(v)
        self.nb_train, self.nb_test, self.nb_val, self.nb_class = get_dataset_sizes(
            train_ids, test_ids, self.y, self.v.train_val_split_ratio, no_val=False
        )

        self.label = configure_labels(self.y, self.nb_train, self.nb_val, self.nb_test)
        set_seed(seed_no)

    def __call__(self):
        for path in self.all_paths:
            self.v.gcn_path = path
            A1, A2, A3, input_type, self.nfeat = get_path(
                self.v, self.FF, self.NF, self.FN, self.NN
            )

            if self.v.gcn_path == "NF-FN-NF":
                self.v.n_hidden.append(100)
                self.A_s = (
                    A1.to(self.gpu),
                    A2.to(self.gpu),
                    A3.to(self.gpu),
                )  # type: ignore
            else:
                self.A_s = (A1.to(self.gpu), A2.to(self.gpu), A3)

            if self.gcn_type != 4:
                self.input_embeddings = get_input_embeddings(
                    input_type, self.gpu, self.A_s, self.v
                )

            self.helper1()
            self.helper2()
            self.helper3()

            self.gcn_model, avg_time = self.gcn_trainer.train_val_loop()
            self.gcn_model.load_state_dict(torch.load(self.model_path))
            # GCN Trainer tek loop yaparsan burası kalkar
            if self.gcn_type == 4:
                with th.no_grad():
                    _, test_w_f1, test_macro, test_micro, test_acc = self.gcn_trainer.loop(  # type: ignore
                        phase="test"
                    )
            else:
                _, test_w_f1, test_macro, test_micro, test_acc = self.gcn_trainer.eval_model(  # type: ignore
                    phase="test"
                )

            print("Test weighted f1 is: {:.3f}".format(100 * test_w_f1))
            print("Test acc is: {:.3f}\n".format(100 * test_acc))

            # acc and micro f1 same thing for multiclass classification
            return test_acc, test_w_f1, avg_time

    def helper1(self):
        print("Type {} Training".format(self.gcn_type))
        if self.gcn_type == 1 or self.gcn_type == 2:
            self.gcn_model = GCN_type1(
                self.A_s, self.nfeat, self.v, self.gpu, nclass=self.nb_class
            )
            self.criterion = nn.CrossEntropyLoss()
        elif self.gcn_type == 3:
            cls_logit = torch.load(
                "bert-finetune_models/{}_logits.pt".format(self.v.dataset)
            )
            self.gcn_model = GCN_type2(
                self.A_s,
                self.nfeat,
                self.v,
                self.gpu,
                cls_logit.to(self.gpu),
                n_class=self.nb_class,
            )
            self.criterion = nn.NLLLoss()
        else:
            self.gcn_model = GCN_type3(
                self.A_s, self.nfeat, self.v, self.v_bert, self.gpu, self.nb_class
            )
            self.criterion = nn.CrossEntropyLoss()

    def helper2(self):
        if self.gcn_type == 4:
            self.optimizer = th.optim.Adam(
                [
                    {
                        "params": self.gcn_model.bert.bert_model.parameters(),  # type: ignore
                        "lr": self.v_bert.lr,
                    },
                    {
                        "params": self.gcn_model.bert.classifier.parameters(),  # type: ignore
                        "lr": self.v_bert.lr,
                    },
                    {
                        "params": self.gcn_model.gcn.parameters(),  # type: ignore
                        "lr": self.v.lr,
                    },  # type: ignore
                ],
                lr=1e-3,
            )  # type: ignore
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[30], gamma=0.1
            )

        else:
            self.optimizer = th.optim.Adam(self.gcn_model.parameters(), lr=self.v.lr)
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[30], gamma=0.1
            )

    def helper3(self):
        self.model_path = "gcn_models/{}_type{}_weights_{}.pt".format(
            self.v.dataset, str(self.gcn_type), self.v.gcn_path
        )
        if self.gcn_type == 4:
            input_ids_, attention_mask_ = encode_input(
                self.v_bert.max_length, list(self.docs), self.gcn_model.bert.tokenizer
            )  # type: ignore

            self.loader, self.dataset_sizes, self.label = configure_bert_inputs(
                input_ids_,
                attention_mask_,
                self.y,
                self.nb_train,
                self.nb_val,
                self.nb_test,
                self.v_bert,
            )

            self.gcn_trainer = BertTrainer(
                self.v,
                self.v_bert,
                self.gpu,
                self.cpu,
                self.gcn_model,
                self.optimizer,
                self.scheduler,
                self.criterion,
                self.loader,
                self.dataset_sizes,
                self.label,
                model_type="gcn",
            )
        else:
            self.gcn_trainer = GCN_Trainer(
                self.gcn_model,
                self.optimizer,
                self.scheduler,
                self.label,
                self.input_embeddings.to(self.gpu),
                self.nb_train,
                self.nb_val,
                self.nb_test,
                self.v,
                self.gpu,
                self.criterion,
                model_path=self.model_path,
            )


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
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
        """Saves model when validation acc increase."""
        if self.verbose:
            self.trace_func(
                f"Validation acc increased ({100*self.val_acc_min:.3f} --> {100*val_acc:.3f}).  Saving model ..."
            )
        th.save(model.state_dict(), self.path)
        self.val_acc_min = val_acc
