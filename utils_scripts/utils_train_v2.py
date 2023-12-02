# %%
from utils_scripts.utils_v2 import *
from model_scripts.trainers import *
from transformers import logging as lg
from scipy import stats

lg.set_verbosity_error()


def set_best_v_bert(v_bert, trial):
    v_bert.lr = trial.params["bert_lr"]
    v_bert.batch_size = trial.params["batch_size"]
    v_bert.max_length = trial.params["max_length"]
    v_bert.bert_init = trial.params["bert_init"]
    v_bert.patience = trial.param["patience"]
    return v_bert


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
        v_bert.bert_lr = trial.params["bert_lr"]
        v_bert.batch_size = trial.params["batch_size"]
        v.m = trial.params["m"]

    return v, v_bert


def get_results_dict(
    results_dic, model_type, best_params, all_paths, v, v_bert, gpu, cpu
):
    final_results = {"test_accs": [], "test_w_f1": []}
    seed_nos = [30, 50, 51, 31, 42, 43, 44, 45, 33, 46]
    for i in range(10):
        tt = Type_Trainer(
            v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type, seed_no=seed_nos[i]
        )
        test_acc, test_w_f1, avg_time = tt()
        final_results["test_accs"].append(test_acc)
        final_results["test_w_f1"].append(test_w_f1)

    gcn_path = get_path_name(all_paths[0])
    method = "Type " + str(model_type) + " " + gcn_path

    test_w_f1_mean = round(100 * np.mean(final_results["test_w_f1"]), 3)
    test_w_f1_std = round(np.std(100 * final_results["test_w_f1"]), 4)
    test_acc_mean = round(100 * np.mean(final_results["test_accs"]), 3)
    test_acc_std = round(np.std(100 * final_results["test_accs"]), 4)

    results_dic["method"].append(method)

    results_dic["test_w_f1"].append((test_w_f1_mean, test_w_f1_std))
    results_dic["test_acc"].append((test_acc_mean, test_acc_std))

    results_dic["best_params"].append(best_params)

    results_dic["avg_time"].append(round(avg_time, 3))
    results_dic["final_results"].append(final_results)

    rv1, rv2 = (
        results_dic["final_results"][-1]["test_accs"],
        results_dic["final_results"][-1]["test_w_f1"],
    )

    p_acc, mean_acc = t_test_maximizer(rv1)
    p_w_f1, mean_w_f1 = t_test_maximizer(rv2)

    results_dic["test_t_acc"].append((round(100 * mean_acc, 4), round(p_acc, 4)))
    results_dic["test_t_w_f1"].append((round(100 * mean_w_f1, 4), round(p_w_f1, 4)))
    return results_dic


def t_test_maximizer(rv1):
    temp = np.mean(rv1)
    p = 1
    eps = 0.000001
    while p > 0.01:
        _, p = stats.ttest_1samp(rv1, popmean=temp)
        temp = temp + eps
        if (p > 0.010) and (p < 0.011):
            return p, temp


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


# df = pd.read_csv("results/R8_type1_2_all_paths.csv")
# df.drop(columns=["Unnamed: 0"], inplace=True)
# df = swap_columns(df, "best_params", "test_acc_mean")
# df = swap_columns(df, "best_params", "test_acc_std")
# df = swap_columns(df, "test_acc_mean", "test_w_f1_mean")
# df = swap_columns(df, "test_acc_std", "test_w_f1_std")
# df.sort_values("model_type", inplace=True)
# df = swap_columns(df, "best_params", "gcn_path")
# df = swap_columns(df, "best_params", "test_t_w_f1")
# df = swap_columns(df, "avg_time", "test_t_acc")


def fix_my_df(df_column):
    for i in range(len(df_column)):
        tmp = df_column[i]
        tmp = eval(tmp)
        t2 = [round(x, 3) for x in tmp]
        a, b = t2
        df_column[i] = (b, a)
