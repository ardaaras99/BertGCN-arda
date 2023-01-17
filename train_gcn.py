# %%
import matplotlib.pyplot as plt
import pandas as pd
from utils_v2 import *
from utils_train_v2 import *
from gcn_models import *
from pathlib import Path
import os
import optuna

# %%

WORK_DIR = Path(__file__).parent  # type: ignore
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)
set_seed()

v.nb_epochs = 1000
v.patience = 50
v.print_gap = 999
# %%
def objective_type1_2_3(trial):
    print("\n*******New Trial Begins*******\n".center(100))
    v.dropout, v.bn_activator = [], []
    n_hidden = [trial.suggest_int("n_hidden", 128, 512, step=32)]
    gcn_lr = trial.suggest_float("gcn_lr", 1e-2, 5e-1, log=True)

    if v.gcn_type == 3:
        m = trial.suggest_float("m", 0.35, 0.85, step=0.05)
        v.m = m
    linear_h = trial.suggest_int("linear_h", 64, 256, step=32)

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
    test_micro, _, _ = tt()
    return test_micro


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


types = [1, 2, 3]
results_dic = {
    "model_type": [],
    "best_params": [],
    "test_acc_mean": [],
    "test_acc_std": [],
    "test_w_f1_mean": [],
    "test_w_f1_std": [],
    "gcn_path": [],
    "avg_time": [],
}
all_pathss = [["FF-NF"], ["FN-NF"], ["NN-NN"], ["NF-NN"]]

for all_paths in all_pathss:
    for model_type in types:
        study = optuna.create_study(direction="maximize")
        txt = "*******GCN type " + str(model_type) + "*******"
        print("\n", txt.center(100), "\n")
        v.gcn_type = model_type
        if v.gcn_type == 1 or v.gcn_type == 2 or v.gcn_type == 3:
            study.optimize(objective_type1_2_3, n_trials=200)  # type: ignore
        elif v.gcn_type == 4:
            study.optimize(objective_type4, n_trials=200)  # type: ignore
        else:
            raise Exception("Invalid GCN type!")

        trial = study.best_trial
        # print("Best Test Micro-F1 {}".format(trial.value))
        # print("Best hyperparameters: {}".format(trial.params))
        best_params = trial.params
        # Retrain with bestparameters
        print("\n", "*******Re-Train with Best Param Setting*******".center(100), "\n")
        v.n_hidden[0] = trial.params["n_hidden"]
        v.lr = trial.params["gcn_lr"]
        v.linear_h = trial.params["linear_h"]
        v.bn_activator[0] = trial.params["bn_activator_0"]
        v.bn_activator[1] = trial.params["bn_activator_1"]
        # v.bn_activator[2] = trial.params["bn_activator_2"]
        v.dropout[0] = trial.params["dropout_0"]
        v.dropout[1] = trial.params["dropout_1"]
        # v.dropout[2] = trial.params["dropout_2"]

        if v.gcn_type == 1 or v.gcn_type == 2 or v.gcn_type == 3:
            v.nb_epochs = 1000
            v.patience = 100
            v.print_gap = 999

        if v.gcn_type == 3:
            v.m = trial.params["m"]

        if v.gcn_type == 4:
            v_bert.bert_lr = trial.params["bert_lrt"]
            v_bert.batch_size = trial.params["batch_size"]
            v.m = trial.params["m"]

        final_results = {"test_accs": [], "test_w_f1": []}

        for i in range(10):
            tt = Type_Trainer(v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type)
            test_acc, test_w_f1, avg_time = tt()
            final_results["test_accs"].append(test_acc)
            final_results["test_w_f1"].append(test_w_f1)

        results_dic["model_type"].append(model_type)
        results_dic["best_params"].append(best_params)
        results_dic["test_acc_mean"].append(100 * np.mean(final_results["test_accs"]))
        results_dic["test_acc_std"].append(np.std(100 * final_results["test_accs"]))
        results_dic["test_w_f1_mean"].append(100 * np.mean(final_results["test_w_f1"]))
        results_dic["test_w_f1_std"].append(np.std(100 * final_results["test_w_f1"]))
        results_dic["gcn_path"].append(all_paths)
        results_dic["avg_time"].append(avg_time)

        # print(
        #     "Avg Test acc with Best Parameter setting is: {:.3f} \u00B1 {:.5f} \n".format(
        #         100 * np.mean(final_results["test_accs"]),
        #         np.std(final_results["test_accs"]),
        #     )
        # )

        # print(
        #     "Avg Test acc with Best Parameter setting is: {:.3f} \u00B1 {:.5f} \n".format(
        #         100 * np.mean(final_results["test_accs"]),
        #         np.std(final_results["test_accs"]),
        #     )
        # )


results_df = pd.DataFrame.from_dict(results_dic)
display(results_df)  # type: ignore

# %%
# EXTRA PLOT SECTION
# x = [pow(10, 0), pow(10, 1.6), pow(10, 4.2),
#      pow(10, 3), pow(10, 3.2), pow(10, 2.7), pow(10, 0.7)]
# y = [86.5, 76, 89.2, 89.4, 89.7, 89.5, 88.8]
# s = ['TypeI', 'TextGCN', 'RBGAT', 'RB', 'RBGCN', 'TypeIII', 'TypeII']
# markers = ["d", "v", "s", "*", "^", "d", "s"]


# fig, ax = plt.subplots(figsize=(15, 5))

# for xp, yp, m in zip(x, y, markers):
#     ax.scatter(xp, yp, marker=m, s=50)  # type: ignore
#     ax.set_xscale('log')

# for i, txt in enumerate(s):
#     ax.annotate(txt, (x[i], y[i]))


# plt.ylabel('Test Accuracy (%)')
# plt.xlabel('Relative Training Time')
# plt.title('Dataset: MR')
# plt.grid(True)
# plt.show()

# model = bert_model
# model.load_state_dict(torch.load(
#     'bert-finetune_models/{}_weights.pt'.format(v.dataset)))


# def eval_model(model, loader, gpu, phase):
#     test_accs, test_losses = [], []
#     with th.no_grad():
#         model.to(gpu)
#         model.eval()
#         for iis, att_mask, lbls in loader[phase]:

#             iis, att_mask, lbls = iis.to(
#                 gpu), att_mask.to(gpu), lbls.to(gpu)

#             y_pred = model(iis, att_mask)
#             y_true = lbls.type(th.long)
#             loss = F.cross_entropy(y_pred, y_true)
#             test_acc = accuracy_score(
#                 y_true.detach().cpu(), y_pred.argmax(axis=1).detach().cpu())

#             # append normalized scores
#             norm_c = y_pred.shape[0] / dataset_sizes[phase][0]
#             test_losses.append(loss.item()*norm_c)
#             test_accs.append(test_acc*norm_c)

#         test_acc = np.sum(test_accs)
#         test_loss = np.sum(test_losses)
#     return test_acc, test_loss


# test_acc, test_loss = eval_model(model, loader, gpu, phase='test')
# print("Test accuracy is: {:.3f}".format(100*test_acc))
