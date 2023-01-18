# %%
from utils_v2 import *
from gcn_models import *
from transformers import logging as lg

lg.set_verbosity_error()


def set_v(v, v_bert, trial):

    v.n_hidden[0] = trial.params["n_hidden"]
    v.lr = trial.params["gcn_lr"]
    v.linear_h = trial.params["linear_h"]
    v.bn_activator[0] = trial.params["bn_activator_0"]
    v.bn_activator[1] = trial.params["bn_activator_1"]
    v.dropout[0] = trial.params["dropout_0"]
    v.dropout[1] = trial.params["dropout_1"]

    if v.gcn_type == 3:
        v.m = trial.params["m"]

    if v.gcn_type == 4:
        v_bert.bert_lr = trial.params["bert_lrt"]
        v_bert.batch_size = trial.params["batch_size"]
        v.m = trial.params["m"]


def get_results_dict(
    results_dic, model_type, best_params, all_paths, v, v_bert, gpu, cpu
):
    final_results = {"test_accs": [], "test_w_f1": []}
    seed_nos = [30, 50, 51, 31, 52, 53, 54, 55, 33, 34]
    for i in range(10):
        tt = Type_Trainer(
            v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type, seed_no=seed_nos[i]
        )
        test_acc, test_w_f1, avg_time = tt()
        final_results["test_accs"].append(test_acc)
        final_results["test_w_f1"].append(test_w_f1)

    results_dic["model_type"].append(model_type)
    results_dic["best_params"].append(best_params)
    results_dic["test_acc_mean"].append(100 * np.mean(final_results["test_accs"]))
    results_dic["test_acc_std"].append(np.std(100 * final_results["test_accs"]))
    results_dic["test_w_f1_mean"].append(100 * np.mean(final_results["test_w_f1"]))
    results_dic["test_w_f1_std"].append(np.std(100 * final_results["test_w_f1"]))
    results_dic["gcn_path"].append(all_paths[0])
    results_dic["avg_time"].append(avg_time)

    return results_dic
