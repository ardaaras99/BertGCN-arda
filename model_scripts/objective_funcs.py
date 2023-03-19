# %%
from model_scripts.trainers import *


class BertObjective:
    def __init__(self, d, v):
        self.d = d
        self.v = v

    def __call__(self, trial):
        self.v.bert_lr = trial.suggest_float(
            "bert_lr", self.d["bert_lr"][0], self.d["bert_lr"][1], log=True
        )
        self.v.batch_size = trial.suggest_categorical(
            "batch_size", self.d["batch_size"]
        )
        self.v.max_length = trial.suggest_categorical(
            "max_length", self.d["max_length"]
        )
        self.v.bert_init = trial.suggest_categorical("bert_init", self.d["bert_init"])
        self.v.patience = trial.suggest_categorical("patience", self.d["patience"])
        bert_trainer = BertTrainer(configure_bert_trainer_inputs(self.v))
        bert_trainer.train_val_loop()

        return round(100 * bert_trainer.metrics["test_acc"], 4)


class GCN_Objective:
    def __init__(self, d, v):
        self.d = d
        self.v = v
        self.v.dropout, self.v.bn_activator = [], []

    def __call__(self, trial):
        # min->max->step
        self.v.n_hidden = trial.suggest_int(
            "n_hidden",
            self.d["n_hidden"][0],
            self.d["n_hidden"][1],
            step=self.d["n_hidden"][2],
        )

        self.v.linear_h = trial.suggest_int(
            "linear_h",
            self.d["linear_h"][0],
            self.d["linear_h"][1],
            step=self.d["linear_h"][2],
        )

        self.v.gcn_lr = trial.suggest_float(
            "gcn_lr", self.d["gcn_lr"][0], self.d["gcn_lr"][1], log=True
        )
        # min->max->step
        if self.v.gcn_type == 3:
            self.v.m = trial.suggest_float(
                "m", self.d["m"][0], self.d["m"][1], step=self.d["m"][2]
            )

        for i in range(2):  # n_hidden otomatik 1 oluyor, 1+1 = 2 diye değiştirdik
            self.v.bn_activator.append(
                trial.suggest_categorical(
                    "bn_activator_{}".format(i), self.d["bn_activator"]
                )
            )
            self.v.dropout.append(
                trial.suggest_float(
                    "dropout_{}".format(i),
                    self.d["dropout"][0],
                    self.d["dropout"][1],
                    step=self.d["dropout"][2],
                )
            )

        tt = Type_Trainer(self.v)
        test_acc, _, _ = tt()
        return round(100 * test_acc, 4)


class Type4_Objective:
    def __init__(self, d, v):
        self.d = d
        self.v = v
        self.v.dropout, self.v.bn_activator = [], []

    def __call__(self, trial):
        # GCN
        self.v.n_hidden = trial.suggest_int(
            "n_hidden",
            self.d["n_hidden"][0],
            self.d["n_hidden"][1],
            step=self.d["n_hidden"][2],
        )

        self.v.gcn_lr = trial.suggest_float(
            "gcn_lr", self.d["gcn_lr"][0], self.d["gcn_lr"][1], log=True
        )

        self.v.m = trial.suggest_float(
            "m", self.d["m"][0], self.d["m"][1], step=self.d["m"][2]
        )

        for i in range(len(self.v.n_hidden) + 1):
            self.v.bn_activator.append(
                trial.suggest_categorical(
                    "bn_activator_{}".format(i), self.d["bn_activator"]
                )
            )
            self.v.dropout.append(
                trial.suggest_float(
                    "dropout_{}".format(i),
                    self.d["dropout"][0],
                    self.d["dropout"][1],
                    step=self.d["dropout"][2],
                )
            )

        # BERT

        self.v.bert_lr = trial.suggest_float(
            "bert_lr", self.d["bert_lr"][0], self.d["bert_lr"][1], log=True
        )
        self.v.batch_size = trial.suggest_categorical(
            "batch_size", self.d["batch_size"]
        )
        self.v.max_length = trial.suggest_categorical(
            "max_length", self.d["max_length"]
        )
        self.v.bert_init = trial.suggest_categorical("bert_init", self.d["bert_init"])
        tt = Type_Trainer(self.v)
        test_micro, _, _ = tt()
        return test_micro
