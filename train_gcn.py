# %%
import matplotlib.pyplot as plt
import pandas as pd

from utils_scripts.utils_v2 import *
from model_scripts.trainers import *
from utils_scripts.utils_train_v2 import *

from pathlib import Path
import os
import optuna
from datetime import datetime

# %%

WORK_DIR = Path(__file__).parent  # type: ignore
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)

v.nb_epochs = 10000
v.patience = 40
v.print_gap = 10
# %%
def objective_type1_2_3(trial):
    print("\n*******New Trial Begins*******\n".center(100))
    v.dropout, v.bn_activator = [], []
    n_hidden = [trial.suggest_int("n_hidden", 64, 256, step=32)]
    gcn_lr = trial.suggest_float("gcn_lr", 1e-4, 5e-1, log=True)

    if v.gcn_type == 3:
        m = trial.suggest_float("m", 0.10, 0.85, step=0.05)
        v.m = m
    linear_h = trial.suggest_int("linear_h", 64, n_hidden[-1], step=32)

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
    test_micro, _, _ = tt()
    return test_micro


current_time = datetime.now().strftime("%d_%m_%_H:%M:%S")
results_path = "results/{}_results_{}.csv".format(v.dataset, current_time)

results_dic = {
    "test_w_f1": [],
    "test_acc": [],
    "method": [],
    "test_t_acc": [],
    "test_t_w_f1": [],
    "avg_time": [],
    "best_params": [],
    "final_results": [],
}
#%%
types = [2, 1]
all_pathss = [["FF-NF"], ["FN-NF"], ["NN-NN"], ["NF-NN"]]
for all_paths in all_pathss:
    for model_type in types:
        study = optuna.create_study(direction="maximize")
        txt = "*******Type " + str(model_type) + " " + all_paths[0] + "*******"
        print("\n", txt.center(100), "\n")
        v.gcn_type = model_type
        if v.gcn_type == 1 or v.gcn_type == 2 or v.gcn_type == 3:
            study.optimize(objective_type1_2_3, n_trials=40)  # type: ignore
        elif v.gcn_type == 4:
            study.optimize(objective_type4, n_trials=1)  # type: ignore
        else:
            raise Exception("Invalid GCN type!")

        trial, best_params = study.best_trial, study.best_trial.params
        print("\n", "*******Re-Train with Best Param Setting*******".center(100), "\n")
        v, v_bert = set_v(v, v_bert, trial)

        v.patience = 100
        v.print_gap = 20
        results_dic = get_results_dict(
            results_dic, model_type, best_params, all_paths, v, v_bert, gpu, cpu
        )

results_df = pd.DataFrame.from_dict(results_dic)
display(results_df)  # type: ignore
results_df.to_csv(results_path)

#%%

# Trial Section for type 3
model_type = 3
all_paths = ["FN-NF"]
v.gcn_type = model_type
v.nb_epochs = 4000
v.patience = 40
v.print_gap = 10

v.n_hidden[0] = 192
v.lr = 0.0026691229074173573
v.m = 0.3
v.linear_h = 160
v.bn_activator[0] = "True"
v.bn_activator[1] = "True"
v.dropout[0] = 0
v.dropout[1] = 0

best_params = 0
results_dic = get_results_dict(
    results_dic, model_type, best_params, all_paths, v, v_bert, gpu, cpu
)
results_df = pd.DataFrame.from_dict(results_dic)
display(results_df)  # type: ignore

# %%
def objective_type4(trial):
    v.dropout, v.bn_activator = [], []

    # GCN
    n_hidden = [trial.suggest_int("n_hidden", 128, 512, step=32)]
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
    test_micro, _, _ = tt()
    return test_micro
