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

# region
nb_epochs = 100

max_length = 2
batch_size = 512
m = 0.7  # lambda in paper
bert_init = 'roberta-base'
pretrained_bert_ckpt = None
dataset = 'mr'
checkpoint_dir = None
gcn_model = 'gcn_sparse'
#gcn_layers = 2
n_hidden = 200
bert_lr = 1e-3
# endregion

# region
no_cuda, fastmode, seed, epochs, gcn_lr = False, False, 42, 200, 2e-3
weight_decay, dropout = 5e-4, 0.5
# endregion

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

_, _, _, adj_nf, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    dataset)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

model = BertGCN_sparse(nfeat=768, nb_class=nb_class, pretrained_model=bert_init, m=m,
                       n_hidden=n_hidden, dropout=dropout)


# region
pretrained_bert_ckpt = "./checkpoint/roberta-base_mr"
if pretrained_bert_ckpt is not None:
    print("sa geldim")
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
# endregion

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)  # has shape (nb_node,)
doc_mask = train_mask + val_mask + test_mask

NF = adj_nf
FN = adj_nf.T  # this one did not work, getting from BertGCN is not correct
NF = normalize_sparse_graph(NF, -0.5, -0.5)
FN = normalize_sparse_graph(FN, -0.5, -0.5)
NF = to_torch_sparse_tensor(NF)
FN = to_torch_sparse_tensor(FN)

data = pd.read_pickle(os.path.join('mr.pkl'))

train_labels, val_labels, test_labels = data['train_labels'], data['val_labels'], data['test_labels']
idx_train, idx_val, idx_test = th.LongTensor(data['train_nodes']), th.LongTensor(
    data['val_nodes']), th.LongTensor(data['test_nodes'])

labels = th.LongTensor(y[doc_mask]).cuda()

dataloader = Data.DataLoader(
    Data.TensorDataset(input_ids[doc_mask],
                       attention_mask[doc_mask]),
    batch_size=1024
)
features = get_bert_output(dataloader, model, gpu)

#features = th.eye(labels.shape[0])

'''
    here I implement HeteGCN(TX-X) or FN-NF path
'''
model_gcn_scratch = GCN_scratch(nfeat=features.shape[1],
                                n_hidden=n_hidden,
                                nclass=labels.max().item() + 1,
                                dropout=dropout)
optimizer = th.optim.Adam(model_gcn_scratch.parameters(),
                          lr=gcn_lr, weight_decay=weight_decay)
# %%
'''
    put in gpu
'''
model_gcn_scratch.cuda()
features = features.cuda()
NF, FN = NF.cuda(), FN.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model_gcn_scratch.train()
    optimizer.zero_grad()
    output = model_gcn_scratch(features, NF, FN)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train, macro_train, micro_train = get_metrics(output[idx_train],
                                                      labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model_gcn_scratch.eval()
        output = model_gcn_scratch(features, NF, FN)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])

    acc_val, macro_val, micro_val = get_metrics(output[idx_val],
                                                labels[idx_val])
    if epoch % (nb_epochs / 20) == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              # 'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'macro_f1_train: {:.4f}'.format(macro_train.item()),
              'micro_f1_train: {:.4f}'.format(micro_train.item()),
              # 'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'macro_f1_val: {:.4f}'.format(macro_val.item()),
              'macro_f1_val: {:.4f}'.format(micro_val.item()))
        # 'time: {:.4f}s'.format(time.time() - t))
    return output


def test():
    model_gcn_scratch.eval()
    output = model_gcn_scratch(features, NF, FN)
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
    output = train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
