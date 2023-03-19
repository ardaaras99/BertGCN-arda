# %%
import pandas as pd
from utils_scripts.utils_v2 import *
from model_scripts.trainers import *
from model_scripts.objective_funcs import *

from pathlib import Path
import os
from datetime import datetime
import optuna

WORK_DIR = Path(__file__).parent  # type: ignore
cur_dir = os.path.basename(__file__)
v, _, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)
v.cpu = cpu
v.gpu = gpu
v.dataset = "Ohsumed"


def run_and_save_gcn_study(param_dict, v, n_trials=1, v_bert=None):

    study = optuna.create_study(study_name=v.current_time, direction="maximize")
    txt = "*******Type " + str(v.gcn_type) + " " + v.gcn_path + "*******"
    print("\n", txt.center(100), "\n")

    if v.gcn_type == 1 or v.gcn_type == 2 or v.gcn_type == 3:
        study.optimize(GCN_Objective(param_dict, v), n_trials=n_trials)
        save_gcn_study(v, study)
    else:
        raise Exception("Invalid GCN type!")


# to generate their results
# v.model_name = "seed-no={}".format(seed_nos[i])
# v.model_path = "hetegcn-results-generated/{}/{}".format(v.dataset, v.model_name)

#%%
# 1.) Hyperparameter search
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

v.nb_epochs = 1000
v.patience = 25
v.print_gap = 25
# bu ikisi save etmediğimiz için önemsiz
v.es_verbose = False
v.model_path = ""
# tüm parametreler her settingde kullanılmıyor bunu hallet
param_dict = {
    "gcn_lr": [1e-4, 1e-1],
    "n_hidden": [64, 512, 32],  # min,max,step
    "linear_h": [64, 512, 32],  # min,max,step
    "m": [0.35, 0.85, 0.05],  # min,max,step
    "bn_activator": ["True"],
    "dropout": [0.2, 0.95, 0.05],
}

types = [3]
paths = ["FF-NF", "FN-NF", "NF-NN", "NN-NN"]
paths = ["FF-NF"]
for gcn_path in paths:
    for gcn_type in types:
        v.gcn_type, v.gcn_path, v.current_time = gcn_type, gcn_path, current_time
        run_and_save_gcn_study(param_dict, v, n_trials=30)


#%%
# 2.) Find best study for all path-type combination and re-train with bestparams
def get_results_dict(results_dict, v):
    final_results = {"test_accs": [], "test_w_f1": []}
    seed_nos = [42, 50, 51, 31, 30, 43, 44, 45, 33, 46]
    for i in range(len(seed_nos)):
        v.model_name = "{}_seed-no={}".format(v.best_study.study_name, seed_nos[i])
        v.model_path = "results/{}/best-gcn-model/type{}/{}/{}".format(
            v.dataset, v.gcn_type, v.gcn_path, v.model_name
        )
        v.es_verbose = True
        tt = Type_Trainer(v, seed_no=seed_nos[i])
        test_acc, test_w_f1, avg_time = tt()

        final_results["test_accs"].append(test_acc)
        final_results["test_w_f1"].append(test_w_f1)
    method = f"Type{v.gcn_type} {v.gcn_path}"
    w_f1, acc = get_mean_test_results(final_results)
    results_dict = update_results_dict(
        results_dict, method, w_f1, acc, v, avg_time, final_results
    )

    return results_dict


v.nb_epochs = 1000
v.patience = 50
v.print_gap = 25
# yukarı şuana kadar ki best
types = [3]
paths = ["FF-NF", "FN-NF", "NF-NN", "NN-NN"]
paths = ["FF-NF"]
paths = ["FN-NF", "NF-NN", "NN-NN"]


results_dict = get_empty_results_dict()
for gcn_path in paths:
    for gcn_type in types:
        v.gcn_type, v.gcn_path = gcn_type, gcn_path
        v.best_study = find_best_study(get_study_path(v))
        v = set_best_v(v, v.best_study.best_trial)
        results_dict = get_results_dict(results_dict, v)

results_df = pd.DataFrame.from_dict(results_dict)
display(results_df)  # type: ignore
#%%
# BURADAKI NAMING HALA MANUAL
# zamana göre değişen bir şey koydum şimdilik en azından
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
project_path = "/Users/ardaaras/Desktop/projects/BertGCN-arda"
print(project_path)
results_path = os.path.join(
    project_path,
    "results/{}/test-results-table/{}.csv".format(v.dataset, current_time),
)
results_df.to_csv(results_path)
# %%
# tüm parametreler her settingde kullanılmıyor bunu hallet
# param_dict = {
#     "gcn_lr": [5e-2, 1e-1],
#     "n_hidden": [64, 256, 32],  # min,max,step
#     "m": [0.35, 0.85, 0.05],  # min,max,step
#     "bn_activator": ["True"],
#     "dropout": [0, 1, 0.05],  # min,max,step
#     "bert_lr": [5e-4, 1e-3],
#     "batch_size": [64, 128],
#     "max_length": [200],
#     "bert_init": ["roberta-base"],
# }

# for type4, we can only pass the following graphs (since document input)
# paths = ["FN-NF", "NN-NN"]

# get results dict functionı içindeydi sonradan ihtiyaç olur belki
# rv1, rv2 = (
#     results_dict["final_results"][-1]["test_accs"],
#     results_dict["final_results"][-1]["test_w_f1"],
# )
# p_acc, mean_acc = t_test_maximizer(rv1)
# p_w_f1, mean_w_f1 = t_test_maximizer(rv2)
# results_dict["test_t_acc"].append((round(100 * mean_acc, 4), round(p_acc, 4)))
# results_dict["test_t_w_f1"].append((round(100 * mean_w_f1, 4), round(p_w_f1, 4)))
