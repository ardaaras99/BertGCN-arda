# %%
import matplotlib.pyplot as plt
from utils_v2 import *
from utils_train_v2 import *
from gcn_models import BertClassifier
from pathlib import Path
import os
import torch.nn.functional as F

import optuna
# %%

WORK_DIR = Path(__file__).parent  # type: ignore
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)
docs, y, train_ids, test_ids, NF, FN, NN, FF = load_corpus(v)

nb_train, nb_test, nb_val, nb_class = get_dataset_sizes(train_ids, test_ids, y)

bert_model = BertClassifier(
    pretrained_model=v_bert.bert_init, nb_class=nb_class)

input_ids_, attention_mask_ = encode_input(
    v_bert.max_length, list(docs), bert_model.tokenizer)
datasets, loader, dataset_sizes, input_ids, attention_mask, label = configure_bert_inputs(
    input_ids_, attention_mask_, y, nb_train, nb_val, nb_test, v)

all_paths = ["FF-NF", "FN-NF", "NN-NN", "NF-NN", "NF-FN-NF"]
all_paths = ["FN-NF"]
# for t in [1]:
#     test_micro = type_trainer(all_paths, label, v, FF, NF, FN, NN,
#                               gpu, nb_train, nb_val, nb_test, nb_class, gcn_type=t)

# %%


def objective(trial):
    n_hidden = [trial.suggest_int('n_hidden', 200, 400)]
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    m = trial.suggest_float('n_hidden', 0.0, 1)

    # train val spliti de optimize et

    v.n_hidden = n_hidden
    v.lr = lr
    v.dropout = dropout
    v.m = m
    test_micro = type_trainer(all_paths, label, v, FF, NF, FN, NN,
                              gpu, nb_train, nb_val, nb_test, nb_class, gcn_type=2)

    return test_micro


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  # type: ignore

# %%
trial = study.best_trial
print("Best Test Micro-F1 {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
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
