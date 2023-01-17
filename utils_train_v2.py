# %%
from utils_v2 import *
from gcn_models import *
from transformers import logging as lg

lg.set_verbosity_error()


def objective_type1_2_3(v, v_bert, gpu, cpu, all_paths, trial):
    print("\n*******New Trial Begins*******\n".center(100))
    v.dropout, v.bn_activator = [], []
    n_hidden = [trial.suggest_int("n_hidden", 128, 512, step=32)]
    gcn_lr = trial.suggest_float("gcn_lr", 5e-2, 5e-1, log=True)

    if v.gcn_type == 3:
        m = trial.suggest_float("m", 0.35, 0.35, step=0.00)
        v.m = m
    linear_h = trial.suggest_int("linear_h", 64, 160, step=32)

    # train val spliti de optimize et

    v.n_hidden = n_hidden
    for i in range(len(n_hidden) + 1):
        bn_activator = trial.suggest_categorical(
            "bn_activator_{}".format(i), ["True", "False"]
        )
        dropout = trial.suggest_float("dropout_{}".format(i), 0.0, 1, step=0.05)
        v.dropout.append(dropout)
        v.bn_activator.append(bn_activator)

    # Set GCN Params
    v.linear_h = linear_h
    v.lr = gcn_lr
    tt = Type_Trainer(v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type)
    test_micro, _ = tt()
    return test_micro


def objective_type4(v, v_bert, gpu, cpu, all_paths, trial):
    v.dropout, v.bn_activator = [], []

    # GCN
    n_hidden = [trial.suggest_int("n_hidden", 320, 512, step=32)]
    gcn_lr = trial.suggest_float("gcn_lr", 1e-4, 1e-1, log=True)
    m = trial.suggest_float("m", 0, 1.0, step=0.05)
    linear_h = trial.suggest_int("linear_h", 32, 128, step=32)

    # BERT
    bert_lr = trial.suggest_float("bert_lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    # train val spliti de optimize et

    v.n_hidden = n_hidden
    for i in range(len(n_hidden) + 1):
        bn_activator = trial.suggest_categorical(
            "bn_activator_{}".format(i), ["True", "False"]
        )
        dropout = trial.suggest_float("dropout_{}".format(i), 0.0, 1, step=0.05)
        v.dropout.append(dropout)
        v.bn_activator.append(bn_activator)

    # Set GCN Params
    v.linear_h = linear_h
    v.lr = gcn_lr
    v.m = m

    # Set BERT Params
    v_bert.batch_size = batch_size
    v_bert.lr = bert_lr

    tt = Type_Trainer(v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type)
    test_micro = tt()
    return test_micro
