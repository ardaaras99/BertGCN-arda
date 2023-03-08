import torch
import torch.nn as nn
import time
from random import seed
from torch.optim import lr_scheduler
import torch.nn.functional as F

from utils_scripts.utils_v2 import *
from model_scripts.gcn_types import *

set_seed()


class GCN_Trainer:
    def __init__(self, input_embeddings, v, model_path=""):

        self.input_embeddings = input_embeddings
        self.v = v
        self.model_path = model_path

    def train_model(self):
        self.v.model.train()
        self.v.optimizer.zero_grad()
        y_pred = self.v.model(self.input_embeddings)[: self.v.nb_train]
        y_true = self.v.label["train"].type(torch.long)
        loss = self.v.criterion(y_pred, y_true)
        w_f1, _, _, _ = get_metrics(y_pred, y_true)
        loss.backward()
        self.v.optimizer.step()
        self.v.scheduler.step()
        train_loss = loss.item()
        return train_loss, w_f1

    def eval_model(self, phase):
        with torch.no_grad():
            self.v.model.eval()
            y_pred = self.v.model(self.input_embeddings)
            if phase == "val":
                y_pred = y_pred[self.v.nb_train : self.v.nb_train + self.v.nb_val]
            else:
                y_pred = y_pred[-self.v.nb_test :]
            y_true = self.v.label[phase].type(torch.long)
            loss = self.v.criterion(y_pred, y_true)
            test_loss = loss.item()
            w_f1, _, _, acc = get_metrics(y_pred, y_true)
        return test_loss, w_f1, acc

    def train_val_loop(self):
        early_stopping = EarlyStopping(
            patience=self.v.patience, verbose=False, path=self.model_path
        )
        avg_time = []
        for epoch in range(1, self.v.nb_epochs + 1):
            start = time.time()
            self.v.current_epoch = epoch
            (train_loss, train_w_f1) = self.train_model()
            val_loss, val_w_f1, val_acc = self.eval_model(phase="val")
            print_results(
                self.v, train_w_f1, val_w_f1, val_acc, train_loss, val_loss, epoch
            )
            early_stopping(val_acc, self.v.model)
            end = time.time()
            avg_time.append(end - start)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        avg_time = np.mean(avg_time)
        return self.v.model, avg_time


class BertTrainer:
    def __init__(self, v, model_type="fine_tune"):
        self.v = v
        self.v.model_type = model_type
        if self.v.model_type != "fine_tune":
            self.gcn_input = self.get_gcn_input()
        else:
            self.gcn_input = torch.tensor([0])

    def save_model(self, path):
        torch.save(self.v.model.state_dict(), path)

    def train_model(self):
        self.v.model.train()
        loss_s, w_f1, macro, micro, acc = 0, 0, 0, 0, 0
        for iis, att_mask, lbls, idx in self.v.loader["train"]:
            self.v.optimizer.zero_grad()
            y_true = lbls.type(torch.long)
            y_pred = self.v.model(iis, att_mask, self.gcn_input, idx)
            loss = F.cross_entropy(y_pred, y_true)
            w_f1_t, macro_t, micro_t, acc_t = get_metrics(y_pred, y_true)

            loss.backward(retain_graph=True)
            self.v.optimizer.step()
            self.v.scheduler.step()

            norm_c = y_pred.shape[0] / self.v.dataset_sizes["train"][0]
            loss_s += loss.item() * norm_c
            w_f1 += w_f1_t * norm_c
            macro += macro_t * norm_c
            micro += micro_t * norm_c
            acc += acc_t * norm_c
        return loss_s, w_f1, macro, micro, acc

    def eval_model(self, phase):
        with torch.no_grad():
            loss_s, w_f1, macro, micro, acc = 0, 0, 0, 0, 0
            self.v.model.eval()
            self.v.model
            for iis, att_mask, lbls, idx in self.v.loader[phase]:
                y_true = lbls.type(torch.long)
                y_pred = self.v.model(
                    iis,
                    att_mask,
                    self.gcn_input,
                    idx,
                )
                loss = F.cross_entropy(y_pred, y_true)
                w_f1_t, macro_t, micro_t, acc_t = get_metrics(y_pred, y_true)
                norm_c = y_pred.shape[0] / self.v.dataset_sizes[phase][0]
                loss_s += loss.item() * norm_c
                w_f1 += w_f1_t * norm_c
                macro += macro_t * norm_c
                micro += micro_t * norm_c
                acc += acc_t * norm_c
        return loss_s, w_f1, macro, micro, acc

    def train_val_loop(self):
        early_stopping = EarlyStopping(patience=self.v.patience, verbose=False)
        for epoch in range(1, self.v.nb_epochs + 1):
            train_loss, train_w_f1, _, _, _ = self.train_model()
            with torch.no_grad():
                val_loss, val_w_f1, _, _, val_acc = self.eval_model(phase="val")
                print_results(
                    self.v, train_w_f1, val_w_f1, val_acc, train_loss, val_loss, epoch
                )
            early_stopping(val_acc, self.v.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return self.v.model

    """
    Burada gcn type4 kullanıyoruz
    self.v.model -> GCN_Type3 tekabul ediyor
    self.v.model.bert_clf -> BertClassfier a tekabul ediyor (BERT ve CLF headden oluşuyor)
    self.v.model.bert_clf.bert_model -> üsttekinin BERT kısmına tekabul ediyor

    """

    def get_gcn_input(self):
        with torch.no_grad():
            self.v.model.eval()
            phases = ["train", "val", "test"]
            cls_list = []
            for phase in phases:
                for iis, att_mask, _, _ in self.v.loader[phase]:
                    cls_feats = self.v.model.bert_clf.bert_model(iis, att_mask)[0][:, 0]
                    cls_list.append(cls_feats)
            input_embeddings = torch.cat(cls_list, axis=0)  # type: ignore
        return input_embeddings

    """
    Bunu berti finetune ederken çağırıyoruz
    self.v.model -> BertClassifier a tekabül ediyor
    """

    def generate_embeddings_and_logits(self):
        with torch.no_grad():
            self.v.model.eval()
            phases = ["train", "val", "test"]
            cls_list = []
            cls_logit_list = []
            for phase in phases:
                for iis, att_mask, _, _ in self.v.loader[phase]:
                    cls_feats = self.v.model.bert_model(iis, att_mask)[0][:, 0]
                    cls_logits = self.v.model(iis, att_mask)
                    cls_list.append(cls_feats)
                    cls_logit_list.append(cls_logits)
            input_embeddings = torch.cat(cls_list, axis=0)
            cls_logits = torch.cat(cls_logit_list, axis=0)

        return cls_logits, input_embeddings


class Type_Trainer:
    def __init__(self, v, seed_no=42):
        self.v = v
        self.v = load_corpus(self.v)
        self.v = get_dataset_sizes(self.v, self.v.train_val_split_ratio)
        self.v = configure_labels(self.v)
        set_seed(seed_no)

    def __call__(self):
        A1, A2, A3, self.input_type, self.v.nfeat = get_path(self.v)
        self.A_s = (A1, A2, A3)

        if self.v.gcn_type != 4:
            self.input_embeddings = get_input_embeddings(
                self.input_type, self.A_s, self.v
            )

        self.v.model, self.v.criterion = self.helper1()
        self.v.optimizer, self.v.scheduler = self.helper2()
        self.v.trainer = self.helper3()

        _, avg_time = self.v.trainer.train_val_loop()

        _, test_w_f1, test_acc = self.v.trainer.eval_model(phase="test")

        print("Test weighted f1 is: {:.3f}".format(100 * test_w_f1))
        print("Test acc is: {:.3f}\n".format(100 * test_acc))

        # acc and micro f1 same thing for multiclass classification
        return test_acc, test_w_f1, avg_time

    def helper1(self):
        print("Type {} Training".format(self.v.gcn_type))
        if self.v.gcn_type == 1 or self.v.gcn_type == 2:
            model = GCN_type1(self.A_s, self.v)
            criterion = nn.CrossEntropyLoss()
        elif self.v.gcn_type == 3:
            bert_path = "results/{}/best-bert-model".format(self.v.dataset)
            ext = [
                filename
                for filename in os.listdir(bert_path)
                if filename.endswith("logits.pt")
            ][0]
            cls_logit_path = os.path.join(bert_path, ext)
            cls_logit = torch.load(cls_logit_path)

            model = GCN_type2(self.A_s, cls_logit, self.v)
            criterion = nn.NLLLoss()
        else:
            model = GCN_type3(self.A_s, self.v)
            criterion = nn.CrossEntropyLoss()
        return model, criterion

    def helper2(self):
        if self.v.gcn_type == 4:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.v.model.bert_clf.bert_model.parameters(),
                        "lr": self.v.bert_lr,
                    },
                    {
                        "params": self.v.model.bert_clf.classifier.parameters(),
                        "lr": self.v.bert_lr,
                    },
                    {
                        "params": self.v.model.gcn.parameters(),
                        "lr": self.v.gcn_lr,
                    },  # type: ignore
                ],
                lr=1e-3,
            )  # type: ignore
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

        else:
            optimizer = torch.optim.Adam(self.v.model.parameters(), lr=self.v.gcn_lr)
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 80], gamma=0.1
            )
        return optimizer, scheduler

    def helper3(self):
        if self.v.gcn_type == 4:
            self.v = helper4(self.v)
            trainer = BertTrainer(self.v, model_type="gcn")
        else:
            trainer = GCN_Trainer(self.input_embeddings, self.v, model_path="")
        return trainer


# we use it only for bert-finetune step
def configure_bert_trainer_inputs(v):
    v.model = BertClassifier(pretrained_model=v.bert_init, nb_class=v.nb_class)
    v.optimizer = torch.optim.Adam(v.model.parameters(), lr=v.bert_lr)
    v.scheduler = lr_scheduler.MultiStepLR(v.optimizer, milestones=[30], gamma=0.1)
    v.criterion = nn.CrossEntropyLoss()
    v.input_ids_, v.attention_mask_ = encode_input(
        v.max_length, list(v.docs), v.model.tokenizer
    )

    v.loader, v.dataset_sizes, v.label = configure_bert_inputs(v)
    return v


def helper4(v):
    v.input_ids_, v.attention_mask_ = encode_input(
        v.max_length, list(v.docs), v.model.bert_clf.tokenizer
    )
    v.loader, v.dataset_sizes, v.label = configure_bert_inputs(v)
    return v


def print_results(v, train_w_f1, val_w_f1, val_acc, train_loss, val_loss, epoch):
    epoch_len = len(str(v.nb_epochs))
    if epoch % v.print_gap == 0:
        print_msg = (
            f"[{epoch:>{epoch_len}}/{v.nb_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.3f} "
            + f"valid_loss: {val_loss:.3f} "
            + f"train_w_f1: {100*train_w_f1:.3f} "
            + f"val_w_f1: {100*val_w_f1:.3f} "
            + f"val_acc: {100*val_acc:.3f} "
        )
        print(print_msg)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience,
        verbose=False,
        path="checkpoint.pt",
        delta=0.0000001,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = 0
        self.delta = delta
        self.path = path

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early Stopping")
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        # default false, to be implemented later
        if self.verbose:
            torch.save(model.state_dict(), self.path)
            print(
                f"Validation acc increased ({100*self.val_acc_min:.3f} --> {100*val_acc:.3f})"
            )
        self.val_acc_min = val_acc
