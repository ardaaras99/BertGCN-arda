# %%
from model import GCN_scratch
import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import sys
import logging
from torch.optim import lr_scheduler
from model import BertGCN_sparse, BertGCN_sparse_concat

from types import SimpleNamespace
from pathlib import Path


WORK_DIR = Path(__file__).parent
CONFIG_PATH = Path.joinpath(
    WORK_DIR, "configs/config_train_bert_hete_gcn.json")
config = load_config_json(CONFIG_PATH)

v = SimpleNamespace(**config)  # store v in config

if v.checkpoint_dir == "":
    ckpt_dir = 'checkpoint/{}_{}_{}'.format(
        v.bert_init, v.gcn_model, v.dataset)
else:
    ckpt_dir = v.checkpoint_dir

os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

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

logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

# Data Preprocess
adj, adj_pmi, adj_tfidf, adj_nf, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    v.dataset)


# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# instantiate model according to class number
if v.use_concat == "yes":
    model = BertGCN_sparse_concat(nfeat=768, nb_class=nb_class, pretrained_model=v.bert_init, m=v.m,
                                  n_hidden=v.n_hidden, dropout=v.dropout)
else:
    model = BertGCN_sparse(nfeat=768, nb_class=nb_class, pretrained_model=v.bert_init, m=v.m,
                           n_hidden=v.n_hidden, dropout=v.dropout)

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
with open("configs/config_train_bert_hete_gcn.json", "w") as outfile:
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


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=v.max_length, truncation=True,
                      padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

'''
    wordleri içinden siliyoruz, bunun direkt alış kısmını değiştiririz
'''
y = np.delete(y, np.arange(nb_train+nb_val, nb_train+nb_val+nb_word))
y_train = np.delete(y_train, np.arange(
    nb_train+nb_val, nb_train+nb_val+nb_word))
y_val = np.delete(y_val, np.arange(nb_train+nb_val, nb_train+nb_val+nb_word))

train_mask = np.delete(train_mask, np.arange(
    nb_train+nb_val, nb_train+nb_val+nb_word))
val_mask = np.delete(val_mask, np.arange(
    nb_train+nb_val, nb_train+nb_val+nb_word))
test_mask = np.delete(test_mask, np.arange(
    nb_train+nb_val, nb_train+nb_val+nb_word))
# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask
# %%
# graph creation
NF = adj_nf
FN = adj_nf.T
NF = normalize_sparse_graph(NF, -0.5, -0.5)
FN = normalize_sparse_graph(FN, -0.5, -0.5)
NF = to_torch_sparse_tensor(NF)
FN = to_torch_sparse_tensor(FN)

NF = NF.to(gpu)
FN = FN.to(gpu)


g_label = th.LongTensor(y)
g_train = th.FloatTensor(train_mask)
g_val = th.FloatTensor(val_mask)
g_test = th.FloatTensor(test_mask)
g_label_train = th.LongTensor(y_train)
g_cls_feats = th.zeros((nb_train+nb_val+nb_test, model.feat_dim))
g_input_ids, g_attention_mask = input_ids, attention_mask


def take_to(x):
    global g_label, g_train, g_val, g_test, g_label_train, g_cls_feats, g_input_ids, g_attention_mask
    g_label = g_label.to(x)
    g_train = g_train.to(x)
    g_val = g_val.to(x)
    g_test = g_test.to(x)
    g_label_train = g_label_train.to(x)
    g_cls_feats = g_cls_feats.to(x)
    g_input_ids = g_input_ids.to(x)
    g_attention_mask = g_attention_mask.to(x)


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
idx_loader = Data.DataLoader(doc_idx, batch_size=v.batch_size, shuffle=True)

# %%
# Training


def update_feature():
    global model, g_cls_feats, g_input_ids, g_attention_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g_input_ids,
                           g_attention_mask),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(
                input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    take_to(cpu)
    g_cls_feats = cls_feat
    return g_cls_feats


optimizer = th.optim.Adam([
    {'params': model.bert_model.parameters(), 'lr': v.bert_lr},
    {'params': model.classifier.parameters(), 'lr': v.bert_lr},
    {'params': model.gcn.parameters(), 'lr': v.gcn_lr},
], lr=1e-3, weight_decay=v.weight_decay
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

train_mask_int = th.FloatTensor(train_mask)


def train_step(engine, batch):
    global model, optimizer, g_input_ids, g_attention_mask, NF, FN, g_cls_feats, g_label_train, g_train
    model.train()
    model = model.to(gpu)
    take_to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    # ya bu train mask çok saçma zaten elimizde var neden gerek duyulmuş bir daha anlamadım
    # çok da bir şey değişmiyor sanki global olarak erişebiliriz
    train_mask = g_train[idx].type(th.BoolTensor)
    y_pred = model(g_input_ids,
                   g_attention_mask, g_cls_feats, NF, FN, idx)[train_mask]

    y_true = g_label_train[idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g_cls_feats.detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, optimizer, g_input_ids, g_attention_mask, NF, FN, g_cls_feats
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        take_to(gpu)

        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g_input_ids,
                       g_attention_mask, g_cls_feats, NF, FN, idx)
        y_true = g_label[idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics = {
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc*100, train_nll, val_acc*100, val_nll, test_acc*100, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
g_cls_feats = update_feature()

# %%
trainer.run(idx_loader, max_epochs=v.nb_epochs)

# %%
