# %%
import time
import torch as th
import torch.nn.functional as F
from utils import *
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
from torch.optim import lr_scheduler
from model import BertGCN_sparse, GCN_scratch
from scipy.sparse import vstack, hstack
import pandas as pd
from model.layers import GraphConvolution
import torch.nn as nn

# Model specification initializations
nb_epochs = 100

max_length = 10
batch_size = 256
m = 0.7  # lambda in paper
bert_init = 'roberta-base'
pretrained_bert_ckpt = None
dataset = 'mr'
checkpoint_dir = None
gcn_model = 'gcn_sparse'
# gcn_layers = 2
n_hidden = 200
bert_lr = 1e-3

no_cuda, fastmode, seed, epochs, gcn_lr = False, False, 42, 200, 2e-3
weight_decay, dropout = 5e-4, 0.5

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

# Data Preprocess
_, _, _, adj_nf, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    dataset)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

model = BertGCN_sparse(nfeat=768, nb_class=nb_class, pretrained_model=bert_init, m=m,
                       n_hidden=n_hidden, dropout=dropout)

if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

corpse_file = './data/corpus/' + dataset + '_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True,
                      padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask


'''
    Here input ids for word indices are all zero, same for attention mask
'''
input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word,
                   max_length), dtype=th.long), input_ids[-nb_test:]])

attention_mask = th.cat([attention_mask[:-nb_test], th.zeros(
    (nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])


# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)  # has shape (nb_node,)

# document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

'''
    getting data from HeteGCN code, seems easier but maybe not true
    25/8/2022 1.44 pm: results are bad only predicts label 1
'''
data = pd.read_pickle(os.path.join('mr.pkl'))

# NF = data['NF']
# FN = data['NF'].T
# %%
'''
    here trying getting it from BertGCN
'''
NF = adj_nf
FN = adj_nf.T
NF = normalize_sparse_graph(NF, -0.5, -0.5)
FN = normalize_sparse_graph(FN, -0.5, -0.5)
NF = to_torch_sparse_tensor(NF)
FN = to_torch_sparse_tensor(FN)


'''
    getting labels from HeteGCN data dictionary
'''
train_labels, val_labels, test_labels = data['train_labels'], data['val_labels'], data['test_labels']
idx_train, idx_val, idx_test = th.LongTensor(data['train_nodes']), th.LongTensor(
    data['val_nodes']), th.LongTensor(data['test_nodes'])
# labels_ = np.concatenate((train_labels, val_labels, test_labels))
# turn one hot encoding to integer labeled
labels = th.LongTensor(y[doc_mask])
labels = labels.cuda()

dataloader = Data.DataLoader(
    Data.TensorDataset(input_ids[doc_mask],
                       attention_mask[doc_mask]),
    batch_size=1024
)
full_features = get_bert_output(dataloader, model, gpu)

# %%

'''
    trying implementation for BertGCN_scratch
'''

optimizer = th.optim.Adam([
    {'params': model.bert_model.parameters(), 'lr': bert_lr},
    {'params': model.classifier.parameters(), 'lr': bert_lr},
    {'params': model.gcn.parameters(), 'lr': gcn_lr},
], lr=1e-3
)
model.cuda()
# features = features.cuda()
NF, FN = NF.cuda(), FN.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
full_features = full_features.cuda()


'''
    trying minibatch implementation
'''
idx_train_dataset = Data.TensorDataset(idx_train)
idx_loader_train = Data.DataLoader(idx_train_dataset, batch_size, shuffle=True)


# %%

dataloader = Data.DataLoader(
    Data.TensorDataset(input_ids[doc_mask],
                       attention_mask[doc_mask]),
    batch_size=1024
)


def update_feature():
    global full_features, dataloader
    full_features = get_bert_output(dataloader, model, gpu)
    full_features = full_features.cuda()


def mini_batch_train(epoch, idx_loader_train):
    for i, batch in enumerate(idx_loader_train):
        (idx, ) = [x for x in batch]
        output = train(epoch, idx)
    return output


def train(epoch, idx):
    t = time.time()
    model.train()
    idx_cpu = idx.cpu()
    optimizer.zero_grad()
    output = model(full_features, NF, FN, input_ids.cuda(),
                   attention_mask.cuda(), th.from_numpy(doc_mask).cuda(), idx)

    loss_train = F.nll_loss(output, labels[idx])
    acc_train, macro_train, micro_train = get_metrics(output,
                                                      labels[idx])
    loss_train.backward()
    optimizer.step()
    update_feature()

    if epoch % (nb_epochs / 20) == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              # 'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'macro_f1_train: {:.4f}'.format(macro_train.item()),
              'micro_f1_train: {:.4f}'.format(micro_train.item()))
    return output


def test():
    model.eval()
    output = model(full_features, NF, FN, input_ids.cuda(),
                   attention_mask.cuda(), th.from_numpy(doc_mask).cuda(), idx_test)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test, macro_test, micro_test = get_metrics(output[idx_test],
                                                   labels[idx_test])
    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()),
          "macro_f1= {:.4f}".format(macro_test.item()),
          "micro_f1= {:.4f}".format(micro_test.item()))


# Train model
t_total = time.time()
for epoch in range(nb_epochs):
    output = mini_batch_train(epoch, idx_loader_train)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
