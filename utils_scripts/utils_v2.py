import numpy as np
from sklearn.metrics import f1_score
import pickle as pkl
import json
from pathlib import Path
from types import SimpleNamespace
import torch.utils.data as Data
import torch
import torch as th
from sklearn.metrics import accuracy_score as acc_func
import random
import os
import joblib


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
    cpu = th.device("cpu")

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


def load_corpus(v):
    docs = read_obj("docs", v)
    y = read_obj("y", v)
    train_ids = read_obj("train_ids", v)
    test_ids = read_obj("test_ids", v)
    # print("Loading corpus for {}".format(v.dataset))
    NF = th.load("datas/{}/graphs/{}_NF.pt".format(v.dataset, v.dataset))
    FN = th.load("datas/{}/graphs/{}_FN.pt".format(v.dataset, v.dataset))
    NN = th.load("datas/{}/graphs/{}_NN.pt".format(v.dataset, v.dataset))
    FF = th.load("datas/{}/graphs/{}_FF.pt".format(v.dataset, v.dataset))

    return docs, y, train_ids, test_ids, NF, FN, NN, FF


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


def configure_labels(y, nb_train, nb_val, nb_test):
    y = th.LongTensor((y).argmax(axis=1))
    label = {}
    label["train"], label["val"], label["test"] = (
        y[:nb_train],
        y[nb_train : nb_train + nb_val],
        y[-nb_test:],
    )
    return label


def configure_bert_inputs(input_ids_, attention_mask_, y, nb_train, nb_val, nb_test, v):
    y = th.LongTensor((y).argmax(axis=1))
    label = {}
    label["train"], label["val"], label["test"] = (
        y[:nb_train],
        y[nb_train : nb_train + nb_val],
        y[-nb_test:],
    )

    input_ids, attention_mask = {}, {}

    input_ids["train"], input_ids["val"], input_ids["test"] = (
        input_ids_[:nb_train],
        input_ids_[nb_train : nb_train + nb_val],
        input_ids_[-nb_test:],
    )

    attention_mask["train"], attention_mask["val"], attention_mask["test"] = (
        attention_mask_[:nb_train],
        attention_mask_[nb_train : nb_train + nb_val],
        attention_mask_[-nb_test:],
    )

    indices = {}
    indices["train"] = th.arange(0, nb_train, dtype=th.long)
    indices["val"] = th.arange(nb_train, nb_train + nb_val, dtype=th.long)
    indices["test"] = th.arange(
        nb_train + nb_val, nb_train + nb_val + nb_test, dtype=th.long
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


def get_dataset_sizes(train_ids, test_ids, y, train_val_split_ratio=0.2, no_val=False):
    if no_val:
        nb_val = 1
    else:
        nb_val = int(train_val_split_ratio * len(train_ids))
    nb_train, nb_test = len(train_ids) - nb_val, len(test_ids)
    nb_class = y.shape[1]
    return nb_train, nb_test, nb_val, nb_class


def get_path(v, FF, NF, FN, NN):
    A3 = 0
    if v.gcn_path == "FF-NF":  # F-X
        A1 = FF
        A2 = NF
        input_type = "word-matrix input"
    elif v.gcn_path == "FN-NF":  # TX-X
        A1 = FN
        A2 = NF
        input_type = "document-matrix input"
    elif v.gcn_path == "NN-NN":  # N-N
        A1 = NN
        A2 = NN
        input_type = "document-matrix input"
    elif v.gcn_path == "NF-NN":  # X-N
        A1 = NF
        A2 = NN
        input_type = "word-matrix input"
    elif v.gcn_path == "NF-FN-NF":  # X-TX-X
        A1 = NF
        A2 = FN
        A3 = NF
        input_type = "word-matrix input"

    if input_type == "word-matrix input":  # type:ignore
        n_feat = A1.shape[1]  # type:ignore
    else:
        if v.gcn_type == 1:
            n_feat = A1.shape[1]  # type:ignore
        else:
            n_feat = 768
    return A1, A2, A3, input_type, n_feat  # type:ignore


def get_input_embeddings(input_type, gpu, A_s, v):
    if input_type == "document-matrix input":
        if v.gcn_type == 1:
            # print("We have input matrix: n_doc x n_doc")
            input_embeddings = th.eye(A_s[0].shape[1]).to(gpu)
        else:
            # print("We have input matrix: n_doc x 768")
            input_embeddings = torch.load(
                "bert-finetune_models/{}_embeddings.pt".format(v.dataset)
            )
    else:
        # print("We have input matrix: n_vocab x n_vocab")
        input_embeddings = th.eye(A_s[0].shape[1]).to(gpu)

    if v.normalize_input_embeddings == "True":
        input_embeddings = max_min_normalize(input_embeddings)

    return input_embeddings


def to_torch_sparse_tensor(M):
    M = M.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((M.row, M.col))).long()
    values = th.from_numpy(M.data)
    shape = th.Size(M.shape)
    T = th.sparse.FloatTensor(indices, values, shape)  # type: ignore
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
