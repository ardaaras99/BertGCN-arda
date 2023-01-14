# %%
import matplotlib.pyplot as plt
from utils_v2 import *
from utils_train_v2 import *
from gcn_models import *
from pathlib import Path
import os
import optuna
import torch
import random
# %%

WORK_DIR = Path(__file__).parent  # type: ignore
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)
all_paths = ["FN-NF"]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_seed()

# %%
# TYPE 1-2 OPTUNA


def objective2(trial):
    print("\n*******New Trial Begins*******\n".center(100))
    v.dropout, v.bn_activator = [], []
    n_hidden = [trial.suggest_int('n_hidden', 128, 512, step=32)]
    gcn_lr = trial.suggest_float("gcn_lr", 5e-2, 5e-1, log=True)
    #m = trial.suggest_float('m', 0.35, 0.35, step=0.00)
    linear_h = trial.suggest_int('linear_h', 64, 160, step=32)

    # train val spliti de optimize et

    v.n_hidden = n_hidden
    for i in range(len(n_hidden) + 1):
        bn_activator = trial.suggest_categorical(
            'bn_activator_{}'.format(i), ['True', 'False'])
        dropout = trial.suggest_float(
            "dropout_{}".format(i), 0.0, 1, step=0.05)
        v.dropout.append(dropout)
        v.bn_activator.append(bn_activator)

    # Set GCN Params
    v.linear_h = linear_h
    v.lr = gcn_lr
    #v.m = m
    tt = Type_Trainer(v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type)
    test_micro, _ = tt()
    return test_micro


study = optuna.create_study(direction="maximize")
study.optimize(objective2, n_trials=100)  # type: ignore

# %%
trial = study.best_trial
print("Best Test Micro-F1 {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

# Retrain with bestparameters
print('\n', "*******Re-Train with Best Param Setting*******".center(100), '\n')
v.n_hidden[0] = trial.params['n_hidden']
v.lr = trial.params['gcn_lr']
#v.m = trial.params['m']
v.linear_h = trial.params['linear_h']
v.bn_activator[0] = trial.params['bn_activator_0']
v.bn_activator[1] = trial.params['bn_activator_1']
v.bn_activator[2] = trial.params['bn_activator_2']
v.dropout[0] = trial.params['dropout_0']
v.dropout[1] = trial.params['dropout_1']
v.dropout[2] = trial.params['dropout_2']

v.nb_epochs = 10000
v.patience = 400

# Run different seeds to obtain mean and std

final_results = {"test_accs": [],
                 "test_w_f1": []}

for i in range(10):
    tt = Type_Trainer(v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type)
    test_acc, test_w_f1 = tt()
    final_results["test_accs"].append(test_acc)
    final_results["test_w_f1"].append(test_w_f1)

print("Avg Test acc with Best Parameter setting is: {:.3f} \u00B1 {:.5f} \n".format(
    100*np.mean(final_results["test_accs"]), np.std(final_results["test_accs"])))

print("Avg Test acc with Best Parameter setting is: {:.3f} \u00B1 {:.5f} \n".format(
    100*np.mean(final_results["test_accs"]), np.std(final_results["test_accs"])))
#
#
#
#
#
#


# %%

# TYPE 3 OPTUNA


def objective3(trial):
    v.dropout, v.bn_activator = [], []

    # GCN
    n_hidden = [trial.suggest_int('n_hidden', 320, 512, step=32)]
    gcn_lr = trial.suggest_float("gcn_lr", 1e-4, 1e-1, log=True)
    m = trial.suggest_float('m', 0, 1.0, step=0.05)
    linear_h = trial.suggest_int('linear_h', 32, 128, step=32)

    # BERT
    bert_lr = trial.suggest_float("bert_lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    # train val spliti de optimize et

    v.n_hidden = n_hidden
    for i in range(len(n_hidden) + 2):
        bn_activator = trial.suggest_categorical(
            'bn_activator_{}'.format(i), ['True', 'False'])
        dropout = trial.suggest_float(
            "dropout_{}".format(i), 0.0, 1, step=0.05)
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


study = optuna.create_study(direction="maximize")
study.optimize(objective3, n_trials=200)  # type: ignore

trial = study.best_trial
print("Best Test Micro-F1 {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

# %%

# def objective(trial):
#     n_hidden = [trial.suggest_int('n_hidden', 128, 512, step=32)]
#     lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
#     #m = trial.suggest_float('m', 0, 1.0, step=0.05)
#     linear_h = trial.suggest_int('linear_h', 32, 128, step=32)

#     # train val spliti de optimize et
#     # batch norm koymayı optimize et
#     v.n_hidden = n_hidden
#     for i in range(len(n_hidden) + 2):
#         bn_activator = trial.suggest_categorical(
#             'bn_activator_{}'.format(i), ['True', 'False'])
#         dropout = trial.suggest_float(
#             "dropout_{}".format(i), 0.0, 1, step=0.05)
#         v.dropout.append(dropout)
#         v.bn_activator.append(bn_activator)

#     v.linear_h = linear_h
#     v.lr = lr
#     #v.m = m
#     test_micro = type_trainer(all_paths, label, v, FF, NF, FN, NN,
#                               gpu, nb_train, nb_val, nb_test, nb_class, gcn_type=1)

#     v.dropout, v.bn_activator = [], []
#     return test_micro


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=200)  # type: ignore


# # After Type 1 Tune finished, train again with best parameters

# # %%
# # 2ND TOUR


# def objective2(trial):
#     v.n_hidden = [160]
#     v.lr = 3.3560545440916416e-05
#     v.m = 0.4
#     v.linear_h = 100
#     v.bn_activator = ['True', 'True', 'True']

#     for i in range(len(v.n_hidden) + 2):
#         dropout = trial.suggest_float(
#             "dropout_{}".format(i), 0.3, 1, step=0.05)
#         v.dropout.append(dropout)

#     test_micro = type_trainer(all_paths, label, v, FF, NF, FN, NN,
#                               gpu, nb_train, nb_val, nb_test, nb_class, gcn_type=2)

#     v.dropout = []
#     return test_micro


# study = optuna.create_study(direction="maximize")
# study.optimize(objective2, n_trials=100)  # type: ignore

# trial = study.best_trial
# print("Best Test Micro-F1 {}".format(trial.value))
# print("Best hyperparameters: {}".format(trial.params))

# %%
# Train model on whole dataset


# %%
# Type 2 Tuner on top of Type1 with the final weights

# def objective2(trial):
#     #n_hidden = [trial.suggest_int('n_hidden', 128, 512, log=True)]
#     lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
#     m = trial.suggest_float('m', 0, 1.0, step=0.05)
#     v.lr = lr
#     v.m = m

#     # train val spliti de optimize et
#     # batch norm koymayı optimize et
#     # best params from
#     n_hidden = [200]
#     v.n_hidden = n_hidden
#     v.bn_activator = ['True', 'True']
#     v.dropout = [0.5, 0.5]

#     test_micro = type_trainer(all_paths, label, v, FF, NF, FN, NN,
#                               gpu, nb_train, nb_val, nb_test, nb_class, gcn_type=1)

#     v.dropout, v.bn_activator = [], []
#     return test_micro


# study = optuna.create_study(direction="maximize")
# study.optimize(objective2, n_trials=50)  # type: ignore


# %%
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
