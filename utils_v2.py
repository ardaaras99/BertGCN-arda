from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp
import pickle as pkl
import json
from pathlib import Path
from types import SimpleNamespace
import torch.utils.data as Data

import torch
import torch as th


def configure_jsons(WORK_DIR, cur_dir):
    CONFIG_PATH_1 = Path.joinpath(
        WORK_DIR, "configs/config_file.json")
    config = load_config_json(CONFIG_PATH_1)

    CONFIG_PATH_2 = Path.joinpath(
        WORK_DIR, "configs/bert_finetune_config.json")
    config2 = load_config_json(CONFIG_PATH_2)

    v = SimpleNamespace(**config)  # store v in config
    v_bert = SimpleNamespace(**config2)  # store v in config
    cpu = th.device('cpu')
    gpu = th.device('cuda:0')

    return v, v_bert, cpu, gpu


def load_config_json(config_path: Path) -> dict:
    with open(config_path) as file:
        return json.load(file)


def read_obj(obj_name, v):

    with open("datas/{}/{}_{}".format(v.dataset, v.dataset, obj_name), 'rb') as f:
        obj = pkl.load(f)
        f.close()
    return obj


def load_corpus(v):
    docs = read_obj('docs', v)
    y = read_obj('y', v)
    train_ids = read_obj('train_ids', v)
    test_ids = read_obj('test_ids', v)
    print("Loading corpus for {}".format(v.dataset))
    NF = th.load('datas/{}/graphs/{}_NF.pt'.format(v.dataset, v.dataset))
    FN = th.load('datas/{}/graphs/{}_FN.pt'.format(v.dataset, v.dataset))
    NN = th.load('datas/{}/graphs/{}_NN.pt'.format(v.dataset, v.dataset))
    FF = th.load('datas/{}/graphs/{}_FF.pt'.format(v.dataset, v.dataset))

    return docs, y, train_ids, test_ids, NF, FN, NN, FF


def make_sym(adj):
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_metrics(output, labels, ID=''):
    preds = output.max(1)[1].type_as(labels)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    """Utility function to compute Accuracy, MicroF1 and Macro F1"""
    # micro f1 ile accuracy aynı multiclass da
    w_f1 = f1_score(y_true, y_pred, average='weighted')
    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')

    return w_f1, macro, micro

# def sparse_to_tuple(sparse_mx):
#     """Convert sparse matrix to tuple representation."""
#     def to_tuple(mx):
#         if not sp.isspmatrix_coo(mx):
#             mx = mx.tocoo()
#         coords = np.vstack((mx.row, mx.col)).transpose()
#         values = mx.data
#         shape = mx.shape
#         return coords, values, shape

#     if isinstance(sparse_mx, list):
#         for i in range(len(sparse_mx)):
#             sparse_mx[i] = to_tuple(sparse_mx[i])
#     else:
#         sparse_mx = to_tuple(sparse_mx)

#     return sparse_mx


# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return sparse_to_tuple(features)


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def preprocess_adj(adj):
#     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
#     return sparse_to_tuple(adj_normalized)


'''
    new utils functions added by Arda Can Aras
'''


def encode_input(max_length, text, tokenizer):
    input = tokenizer(text, max_length=max_length,
                      truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask


def max_min_normalize(x):
    x_normed = (x - x.min(0, keepdim=True)
                [0]) / (x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0])
    return x_normed


def configure_bert_inputs(input_ids_, attention_mask_, y, nb_train, nb_val, nb_test, v):
    y = th.LongTensor((y).argmax(axis=1))
    label = {}
    label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]

    input_ids, attention_mask = {}, {}

    input_ids['train'], input_ids['val'], input_ids['test'] = input_ids_[
        :nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]

    attention_mask['train'], attention_mask['val'], attention_mask['test'] = attention_mask_[
        :nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]

    datasets = {}
    loader = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = Data.TensorDataset(
            input_ids[split], attention_mask[split], label[split])
        loader[split] = Data.DataLoader(
            datasets[split], batch_size=v.batch_size, shuffle=False)

    dataset_sizes = {'train': label['train'].shape,
                     'val': label['val'].shape,
                     'test': label['test'].shape}

    return datasets, loader, dataset_sizes, input_ids, attention_mask, label


def get_dataset_sizes(train_ids, test_ids, y):
    nb_val = int(0.2*len(train_ids))
    nb_train, nb_test = len(train_ids) - nb_val, len(test_ids)
    nb_class = y.shape[1]
    return nb_train, nb_test, nb_val, nb_class


def get_path(v, FF, NF, FN, NN):
    A3 = 0
    if v.gcn_path == 'FF-NF':  # F-X
        A1 = FF
        A2 = NF
        input_type = "word-matrix input"
    elif v.gcn_path == 'FN-NF':  # TX-X
        A1 = FN
        A2 = NF
        input_type = "document-matrix input"
    elif v.gcn_path == 'NN-NN':  # N-N
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
    return A1, A2, A3, input_type  # type:ignore


def get_input_embeddings(input_type, gpu, A_s, v):
    if input_type == "document-matrix input":
        print("We have input matrix: n_doc x 768")
        input_embeddings = torch.load(
            'bert-finetune_models/{}_embeddings.pt'.format(v.dataset))
    else:
        print("We have input matrix: n_vocab x n_vocab")
        input_embeddings = th.eye(A_s[0].shape[1]).to(gpu)

    if v.normalize_input_embeddings == 'True':
        input_embeddings = max_min_normalize(input_embeddings)
    nfeat = input_embeddings.shape[1]
    return input_embeddings, nfeat


def to_torch_sparse_tensor(M):
    M = M.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((M.row, M.col))).long()
    values = th.from_numpy(M.data)
    shape = th.Size(M.shape)
    T = th.sparse.FloatTensor(indices, values, shape)  # type: ignore
    return T
