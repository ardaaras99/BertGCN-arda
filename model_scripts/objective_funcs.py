# %%
from model_scripts.trainers import *


class BertObjective:
    def __init__(self, d, v_bert):
        self.d = d
        self.v_bert = v_bert

    def __call__(self, trial):
        bert_lr = trial.suggest_float(
            "bert_lr", self.d["bert_lr"][0], self.d["bert_lr"][1], log=True
        )
        batch_size = trial.suggest_categorical("batch_size", self.d["batch_size"])
        max_length = trial.suggest_categorical("max_length", self.d["max_length"])
        bert_init = trial.suggest_categorical("bert_init", self.d["bert_init"])
        patience = trial.suggest_categorical("patience", self.d["patience"])

        self.v_bert.lr = bert_lr
        self.v_bert.batch_size = batch_size
        self.v_bert.max_length = max_length
        self.v_bert.bert_init = bert_init
        self.v_bert.patience = patience

        bert_fine_tuner = BertTrainer(*configure_bert_trainer_inputs(self.v_bert))

        bert_fine_tuner.train_val_loop()
        with th.no_grad():
            _, test_w_f1, test_macro, test_micro, test_acc = bert_fine_tuner.loop(
                phase="test"
            )

        print("Test weighted f1 is: {:.3f}".format(100 * test_w_f1))
        print("Test macro f1 is: {:.3f}".format(100 * test_macro))
        print("Test micro f1 is: {:.3f}".format(100 * test_micro))
        print("Test acc is: {:.3f}".format(100 * test_acc))

        return test_micro
