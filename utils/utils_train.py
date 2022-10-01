from utils import*
import shutil
import sys
import logging
import os
from types import SimpleNamespace
from model import BertGCN_sparse, BertGCN_sparse_concat
import torch as th
import torch.utils.data as Data
from transformers import logging as lg
from sklearn.neighbors import kneighbors_graph

lg.set_verbosity_error()


def configure(WORK_DIR):
    CONFIG_PATH = Path.joinpath(
        WORK_DIR, "configs/config_file.json")
    config = load_config_json(CONFIG_PATH)
    v = SimpleNamespace(**config)  # store v in config

    if v.checkpoint_dir == "":
        ckpt_dir = 'checkpoint/{}_{}_{}'.format(
            v.bert_init, v.gcn_model, v.dataset)
    else:
        ckpt_dir = v.checkpoint_dir

    # buraya gerek var mı tunaya sor
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=os.path.join(
        ckpt_dir, 'training.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logger = logging.getLogger('training logger')
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    cpu = th.device('cpu')
    gpu = th.device('cuda:0')

    return v, ckpt_dir, config, sh, fh, logger, cpu, gpu


def set_variables(v, gpu, config):
    # Graph Creation

    adj, adj_pmi, adj_tfidf, adj_nf, adj_ff, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
        v.dataset)

    NF, FN, NN, FF = get_graphs(v, gpu)

    A1, A2, A3, input_type = get_path(v, FF, NF, FN, NN)
    A_s = (A1, A2, A3)
    if input_type == "document-matrix input":
        nfeat = 768
    else:
        nfeat = FF.shape[0]
    # End of Graph Creation

    nb_node = features.shape[0]
    nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    nb_word = nb_node - nb_train - nb_val - nb_test
    nb_class = y_train.shape[1]

    # nfeat is not always 768 we need to check it depending on input type
    if v.use_concat == "yes":
        model = BertGCN_sparse_concat(nfeat=nfeat, nb_class=nb_class, pretrained_model=v.bert_init, m=v.m,
                                      n_hidden=v.n_hidden, dropout=v.dropout, A_s=A_s)
    else:
        model = BertGCN_sparse(input_type, nfeat=nfeat, nb_class=nb_class, pretrained_model=v.bert_init, m=v.m,
                               n_hidden=v.n_hidden, dropout=v.dropout, A_s=A_s)

    if v.pretrained_bert_ckpt != "":
        print("We use pretrained model")
        ckpt = th.load(os.path.join(
            v.pretrained_bert_ckpt, 'checkpoint.pth'
        ), map_location=gpu)
        model.bert_model.load_state_dict(ckpt['bert_model'])
        model.classifier.load_state_dict(ckpt['classifier'])

    config["pretrained_bert_ckpt"] = ""
    # Serializing json
    json_object = json.dumps(config, indent=4)

    # Writing to sample.json
    with open("configs/config_file.json", "w") as outfile:
        outfile.write(json_object)

    # load documents and compute input encodings
    corpse_file = './data/corpus/' + v.dataset + '_shuffle.txt'
    with open(corpse_file, 'r') as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')

    '''
        here we get the maximum sentence lenght and update v.max_length accordingly
    '''
    c_max = max([len(sentence.split()) for sentence in text])

    if c_max < v.max_length:
        v.max_length = c_max
    #print("Max length for bert is " + str(v.max_length))

    input_ids, attention_mask = encode_input(text, model.tokenizer, v)

    y = y_train + y_test + y_val
    y_train = y_train.argmax(axis=1)
    y = y.argmax(axis=1)

    y = np.delete(y, np.arange(nb_train+nb_val, nb_train+nb_val+nb_word))
    y_train = np.delete(y_train, np.arange(
        nb_train+nb_val, nb_train+nb_val+nb_word))
    y_val = np.delete(y_val, np.arange(
        nb_train+nb_val, nb_train+nb_val+nb_word))

    train_mask = np.delete(train_mask, np.arange(
        nb_train+nb_val, nb_train+nb_val+nb_word))
    val_mask = np.delete(val_mask, np.arange(
        nb_train+nb_val, nb_train+nb_val+nb_word))
    test_mask = np.delete(test_mask, np.arange(
        nb_train+nb_val, nb_train+nb_val+nb_word))
    # document mask used for update feature
    doc_mask = train_mask + val_mask + test_mask

    g_label = th.LongTensor(y)
    g_train = th.FloatTensor(train_mask)
    g_val = th.FloatTensor(val_mask)
    g_test = th.FloatTensor(test_mask)
    g_label_train = th.LongTensor(y_train)
    #g_cls_feats = th.zeros((nb_train+nb_val+nb_test, model.feat_dim))
    g_input_ids, g_attention_mask = input_ids, attention_mask

    # create index loader
    train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
    val_idx = Data.TensorDataset(
        th.arange(nb_train, nb_train + nb_val, dtype=th.long))
    test_idx = Data.TensorDataset(
        th.arange(nb_train+nb_val, nb_train+nb_val+nb_test, dtype=th.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    idx_loader_train = Data.DataLoader(
        train_idx, batch_size=v.batch_size, shuffle=True)
    idx_loader_val = Data.DataLoader(val_idx, batch_size=v.batch_size)
    idx_loader_test = Data.DataLoader(test_idx, batch_size=v.batch_size)
    idx_loader = Data.DataLoader(
        doc_idx, batch_size=v.batch_size, shuffle=True)

    optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': v.bert_lr},
        {'params': model.classifier.parameters(), 'lr': v.bert_lr},
        {'params': model.gcn.parameters(), 'lr': v.gcn_lr},
    ], lr=1e-3, weight_decay=v.weight_decay
    )

    return model, optimizer, A_s, input_type, train_mask, g_label, g_train, g_val, g_test, g_label_train, g_input_ids, g_attention_mask, idx_loader_train, idx_loader_val, idx_loader_test, idx_loader


def get_graphs(v, gpu):

    adj, adj_pmi, adj_tfidf, adj_nf, adj_ff, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
        v.dataset)

    NF = adj_nf
    FN = adj_nf.T
    FF = adj_ff
    n_neighbors = 25
    metric = 'cosine'
    NN = kneighbors_graph(NF, n_neighbors,
                          metric=metric, include_self=True)

    NF = normalize_sparse_graph(NF, -0.5, -0.5)
    FN = normalize_sparse_graph(FN, -0.5, -0.5)
    NN = normalize_sparse_graph(NN, -0.5, -0.5)
    FF = normalize_sparse_graph(FF, -0.5, -0.5)

    NN, NF, FN, FF = to_torch_sparse_tensor(NN), to_torch_sparse_tensor(
        NF), to_torch_sparse_tensor(FN), to_torch_sparse_tensor(FF)

    NF, FN, NN, FF = NF.to(gpu), FN.to(gpu), NN.to(gpu), FF.to(gpu)

    return NF, FN, NN, FF


def encode_input(text, tokenizer, v):
    input = tokenizer(text, max_length=v.max_length, truncation=True,
                      padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask


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
    return A1, A2, A3, input_type
