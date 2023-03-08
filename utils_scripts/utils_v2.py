import numpy as np
from sklearn.metrics import f1_score
import pickle as pkl
import json
from pathlib import Path
from types import SimpleNamespace
import torch.utils.data as Data
import torch
from sklearn.metrics import accuracy_score as acc_func
import random
import os
import joblib
from transformers import logging as lg
from scipy import stats

lg.set_verbosity_error()


def set_best_v_bert(v_bert, trial):
    v_bert.lr = trial.params["bert_lr"]
    v_bert.batch_size = trial.params["batch_size"]
    v_bert.max_length = trial.params["max_length"]
    v_bert.bert_init = trial.params["bert_init"]
    v_bert.patience = trial.params["patience"]
    return v_bert


def set_best_v(v, trial):

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
        v.v_bert.bert_lr = trial.params["bert_lr"]
        v.v_bert.batch_size = trial.params["batch_size"]
        v.m = trial.params["m"]

    return v


# def get_results_dict(
#     results_dic, model_type, best_params, all_paths, v, v_bert, gpu, cpu
# ):
#     final_results = {"test_accs": [], "test_w_f1": []}
#     seed_nos = [30, 50, 51, 31, 42, 43, 44, 45, 33, 46]
#     for i in range(10):
#         tt = Type_Trainer(
#             v, v_bert, all_paths, gpu, cpu, gcn_type=v.gcn_type, seed_no=seed_nos[i]
#         )
#         test_acc, test_w_f1, avg_time = tt()
#         final_results["test_accs"].append(test_acc)
#         final_results["test_w_f1"].append(test_w_f1)

#     gcn_path = get_path_name(all_paths[0])
#     method = "Type " + str(model_type) + " " + gcn_path

#     test_w_f1_mean = round(100 * np.mean(final_results["test_w_f1"]), 3)
#     test_w_f1_std = round(np.std(100 * final_results["test_w_f1"]), 4)
#     test_acc_mean = round(100 * np.mean(final_results["test_accs"]), 3)
#     test_acc_std = round(np.std(100 * final_results["test_accs"]), 4)

#     results_dic["method"].append(method)

#     results_dic["test_w_f1"].append((test_w_f1_mean, test_w_f1_std))
#     results_dic["test_acc"].append((test_acc_mean, test_acc_std))

#     results_dic["best_params"].append(best_params)

#     results_dic["avg_time"].append(round(avg_time, 3))
#     results_dic["final_results"].append(final_results)

#     rv1, rv2 = (
#         results_dic["final_results"][-1]["test_accs"],
#         results_dic["final_results"][-1]["test_w_f1"],
#     )

#     p_acc, mean_acc = t_test_maximizer(rv1)
#     p_w_f1, mean_w_f1 = t_test_maximizer(rv2)

#     results_dic["test_t_acc"].append((round(100 * mean_acc, 4), round(p_acc, 4)))
#     results_dic["test_t_w_f1"].append((round(100 * mean_w_f1, 4), round(p_w_f1, 4)))
#     return results_dic


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


def fix_my_df(df_column):
    for i in range(len(df_column)):
        tmp = df_column[i]
        tmp = eval(tmp)
        t2 = [round(x, 3) for x in tmp]
        a, b = t2
        df_column[i] = (b, a)


def get_path_name(tmp):
    if tmp == "FN-NF":
        return "(TX-X)"
    if tmp == "FF-NF":
        return "(F-X)"
    if tmp == "NN-NN":
        return "(N-N)"
    if tmp == "NF-NN":
        return "(X-N)"


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


def configure_jsons(WORK_DIR, cur_dir):
    CONFIG_PATH_1 = Path.joinpath(WORK_DIR, "configs/config_file.json")
    config = load_config_json(CONFIG_PATH_1)

    CONFIG_PATH_2 = Path.joinpath(WORK_DIR, "configs/bert_finetune_config.json")
    config2 = load_config_json(CONFIG_PATH_2)

    v = SimpleNamespace(**config)  # store v in config
    v_bert = SimpleNamespace(**config2)  # store v in config
    cpu = torch.device("cpu")

    if torch.backends.cuda.is_built():
        print("CUDA")
        gpu = torch.device("cuda")
    elif torch.backends.mps.is_built():
        print("mps")
        gpu = torch.device("mps")
    else:
        raise Exception("GPU is not avalaible!")
        # gpu = torch.device("cpu")

    return v, v_bert, cpu, gpu


def load_config_json(config_path: Path) -> dict:
    with open(config_path) as file:
        return json.load(file)


def read_obj(obj_name, v):

    with open("datas/{}/{}_{}".format(v.dataset, v.dataset, obj_name), "rb") as f:
        obj = pkl.load(f)
        f.close()
    return obj


def make_sym(adj):
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def get_metrics(output, labels, ID=""):
    preds = output.max(1)[1].type_as(labels)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    """Utility function to compute Accuracy, MicroF1 and Macro F1"""
    # micro f1 ile accuracy aynı multiclass da
    w_f1 = f1_score(y_true, y_pred, average="weighted")
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    acc = acc_func(y_true, y_pred)
    return w_f1, macro, micro, acc


def encode_input(max_length, text, tokenizer):
    input = tokenizer(
        text, max_length=max_length, truncation=True, padding=True, return_tensors="pt"
    )
    return input.input_ids, input.attention_mask


def max_min_normalize(x):
    x_normed = (x - x.min(0, keepdim=True)[0]) / (
        x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0]
    )
    return x_normed


def load_corpus(v):
    v.docs = read_obj("docs", v)
    v.y = read_obj("y", v)
    v.train_ids = read_obj("train_ids", v)
    v.test_ids = read_obj("test_ids", v)
    # print("Loading corpus for {}".format(v.dataset))
    v.NF = torch.load("datas/{}/graphs/{}_NF.pt".format(v.dataset, v.dataset))
    v.FN = torch.load("datas/{}/graphs/{}_FN.pt".format(v.dataset, v.dataset))
    v.NN = torch.load("datas/{}/graphs/{}_NN.pt".format(v.dataset, v.dataset))
    v.FF = torch.load("datas/{}/graphs/{}_FF.pt".format(v.dataset, v.dataset))

    return v


def get_dataset_sizes(v, train_val_split_ratio=0.2):

    v.nb_val = int(train_val_split_ratio * len(v.train_ids))
    v.nb_train, v.nb_test = len(v.train_ids) - v.nb_val, len(v.test_ids)
    v.nb_class = v.y.shape[1]
    return v


def configure_labels(v):
    y = torch.LongTensor((v.y).argmax(axis=1))
    label = {}
    label["train"], label["val"], label["test"] = (
        y[: v.nb_train],
        y[v.nb_train : v.nb_train + v.nb_val],
        y[-v.nb_test :],
    )
    v.label = label
    return v


def configure_bert_inputs(v):
    y = torch.LongTensor((v.y).argmax(axis=1))
    label = {}
    label["train"], label["val"], label["test"] = (
        y[: v.nb_train],
        y[v.nb_train : v.nb_train + v.nb_val],
        y[-v.nb_test :],
    )

    input_ids, attention_mask = {}, {}

    input_ids["train"], input_ids["val"], input_ids["test"] = (
        v.input_ids_[: v.nb_train],
        v.input_ids_[v.nb_train : v.nb_train + v.nb_val],
        v.input_ids_[-v.nb_test :],
    )

    attention_mask["train"], attention_mask["val"], attention_mask["test"] = (
        v.attention_mask_[: v.nb_train],
        v.attention_mask_[v.nb_train : v.nb_train + v.nb_val],
        v.attention_mask_[-v.nb_test :],
    )

    indices = {}
    indices["train"] = torch.arange(0, v.nb_train, dtype=torch.long)
    indices["val"] = torch.arange(v.nb_train, v.nb_train + v.nb_val, dtype=torch.long)
    indices["test"] = torch.arange(
        v.nb_train + v.nb_val,
        v.nb_train + v.nb_val + v.nb_test,
        dtype=torch.long,
    )

    datasets = {}
    loader = {}
    for split in ["train", "val", "test"]:
        datasets[split] = Data.TensorDataset(
            input_ids[split], attention_mask[split], label[split], indices[split]
        )
        loader[split] = Data.DataLoader(
            datasets[split], batch_size=v.batch_size, shuffle=False
        )

    dataset_sizes = {
        "train": label["train"].shape,
        "val": label["val"].shape,
        "test": label["test"].shape,
    }

    return loader, dataset_sizes, label


def get_path(v):
    A3 = 0
    if v.gcn_path == "FF-NF":  # F-X
        A1 = v.FF
        A2 = v.NF
        input_type = "word-matrix input"
    elif v.gcn_path == "FN-NF":  # TX-X
        A1 = v.FN
        A2 = v.NF
        input_type = "document-matrix input"
    elif v.gcn_path == "NN-NN":  # N-N
        A1 = v.NN
        A2 = v.NN
        input_type = "document-matrix input"
    elif v.gcn_path == "NF-NN":  # X-N
        A1 = v.NF
        A2 = v.NN
        input_type = "word-matrix input"
    elif v.gcn_path == "NF-FN-NF":  # X-TX-X
        A1 = v.NF
        A2 = v.FN
        A3 = v.NF
        input_type = "word-matrix input"

    if input_type == "word-matrix input":  # type:ignore
        n_feat = A1.shape[1]  # type:ignore
    else:
        if v.gcn_type == 1:
            n_feat = A1.shape[1]  # type:ignore
        else:
            n_feat = 768
    return A1, A2, A3, input_type, n_feat  # type:ignore


def get_input_embeddings(input_type, A_s, v):
    if input_type == "document-matrix input":
        if v.gcn_type == 1:
            # print("We have input matrix: n_doc x n_doc")
            input_embeddings = torch.eye(A_s[0].shape[1])
        else:
            # print("We have input matrix: n_doc x 768")
            model_path = "results/{}/best-bert-model".format(v.dataset)
            ext = [
                filename
                for filename in os.listdir(model_path)
                if filename.endswith("embeddings.pt")
            ][0]
            embedding_path = os.path.join(model_path, ext)
            input_embeddings = torch.load(embedding_path)
    else:
        # print("We have input matrix: n_vocab x n_vocab")
        input_embeddings = torch.eye(A_s[0].shape[1])

    if v.normalize_input_embeddings == "True":
        input_embeddings = max_min_normalize(input_embeddings)

    return input_embeddings


def to_torch_sparse_tensor(M):
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    T = torch.sparse.FloatTensor(indices, values, shape)  # type: ignore
    return T


def find_best_study(hyperparamstudy_path):
    best_study = None
    best_val = 0
    for study_name in os.listdir(hyperparamstudy_path):
        cur_study_path = os.path.join(hyperparamstudy_path, study_name)
        study = joblib.load(cur_study_path)
        if study.best_trial.value > best_val:
            best_val = study.best_trial.value
            best_study = study
    return best_study


def save_gcn_study(v, study):
    joblib.dump(
        study,
        "results/{}/hyperparam-studies/type{}/{}/{}.pkl".format(
            v.dataset, v.gcn_type, v.gcn_path, study.study_name
        ),
    )
