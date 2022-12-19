# %%
from utils_v2 import *
from utils_train_v2 import *
from gcn_models import *
from pathlib import Path
import os
from torch.optim import lr_scheduler
import torch
import torch as th
import torch.nn.functional as F

# %%

# DATA LOADING
WORK_DIR = Path(__file__).parent
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)
docs, y, train_ids, test_ids, NF, FN, NN, FF = load_corpus(v)

c_max = max([len(sentence.split()) for sentence in docs])

if c_max < v_bert.max_length:
    v_bert.max_length = c_max
# docs are already shuffled according to ids, the first nb_train part corresponds to train set and etc...

nb_train, nb_test, nb_val, nb_class = get_dataset_sizes(train_ids, test_ids, y)

model = BertClassifier(pretrained_model=v_bert.bert_init, nb_class=nb_class)
input_ids_, attention_mask_ = encode_input(
    v_bert.max_length, list(docs), model.tokenizer)

datasets, loader, dataset_sizes, input_ids, attention_mask, label = configure_bert_inputs(
    input_ids_, attention_mask_, y, nb_train, nb_val, nb_test, v)

# %%
# Specify Loss Functions and Optimizer
optimizer = th.optim.Adam(model.parameters(), lr=v_bert.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

# %%


def train_model(model, loader, gpu, phase='train'):
    train_losses, train_accs = [], []
    model.to(gpu)
    model.train()

    for iis, att_mask, lbls in loader[phase]:

        iis, att_mask, lbls = iis.to(
            gpu), att_mask.to(gpu), lbls.to(gpu)

        optimizer.zero_grad()
        y_true = lbls.type(th.long)
        y_pred = model(iis, att_mask)
        loss = F.cross_entropy(y_pred, y_true)
        train_acc = accuracy_score(
            y_true.detach().cpu(), y_pred.argmax(axis=1).detach().cpu())
        loss.backward()
        optimizer.step()
        scheduler.step()

        # since all batches do not have equal lengths we need to normalize loss and accs by a
        # constant  = batch_size / dataset_size
        norm_c = y_pred.shape[0] / dataset_sizes['train'][0]
        train_losses.append(loss.item() * norm_c)
        train_accs.append(train_acc * norm_c)

    train_loss = np.sum(train_losses)
    train_acc = np.sum(train_accs)

    return train_acc, train_loss


def eval_model(model, loader, gpu, phase):
    test_accs, test_losses = [], []
    with th.no_grad():
        model.to(gpu)
        model.eval()
        for iis, att_mask, lbls in loader[phase]:

            iis, att_mask, lbls = iis.to(
                gpu), att_mask.to(gpu), lbls.to(gpu)

            y_pred = model(iis, att_mask)
            y_true = lbls.type(th.long)
            loss = F.cross_entropy(y_pred, y_true)
            test_acc = accuracy_score(
                y_true.detach().cpu(), y_pred.argmax(axis=1).detach().cpu())

            # append normalized scores
            norm_c = y_pred.shape[0] / dataset_sizes[phase][0]
            test_losses.append(loss.item()*norm_c)
            test_accs.append(test_acc*norm_c)

        test_acc = np.sum(test_accs)
        test_loss = np.sum(test_losses)
    return test_acc, test_loss


def finetune_bert(model, loader, dataset_name, patience, n_epochs, gpu, print_gap):
    model.to(gpu)
    avg_losses = {
        'train': [],
        'val': []
    }

    avg_accs = {
        'train': [],
        'val': []
    }

    early_stopping = EarlyStopping(
        patience=patience, verbose=True,
        path='bert-finetune_models/{}_weights.pt'.format(dataset_name))

    for epoch in range(1, n_epochs+1):

        train_acc, train_loss = train_model(model, loader, gpu, phase='train')

        val_acc, val_loss = eval_model(model, loader, gpu, phase='val')

        # just sum since we already weight them

        avg_losses['train'].append(train_loss)
        avg_losses['val'].append(val_loss)
        avg_accs['train'].append(train_acc)
        avg_accs['val'].append(val_acc)

        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.3f} ' +
                     f'valid_loss: {val_loss:.3f} ' +
                     f'train_acc: {100*train_acc:.3f} ' +
                     f'val_acc: {100*val_acc:.3f} ')

        print(print_msg)

        # monitor val_acc
        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(
        'bert-finetune_models/{}_weights.pt'.format(dataset_name)))

    return model, avg_losses, avg_accs


# %%
# Train
model, avg_loss, avg_accs = finetune_bert(
    model, loader, v_bert.dataset, v_bert.patience, v_bert.nb_epochs, gpu, v_bert.print_gap)

# %%
# Test Model
model.load_state_dict(torch.load(
    'bert-finetune_models/{}_weights.pt'.format(v.dataset)))

test_acc, test_loss = eval_model(model, loader, gpu, phase='test')
print("Test accuracy is: {:.3f}".format(100*test_acc))

# %%
# Save BERT embeddings also
with th.no_grad():
    model.to(gpu)
    model.eval()
    phases = ['train', 'val', 'test']
    cls_list = []
    cls_logit_list = []
    for phase in phases:
        for iis, att_mask, lbls in loader[phase]:
            cls_feats = model.bert_model(
                iis.to(gpu), att_mask.to(gpu))[0][:, 0]  # original bert_model has classifier head we get rid of it
            cls_logits = model(iis.to(gpu), att_mask.to(gpu))

            cls_list.append(cls_feats.to(cpu))
            cls_logit_list.append(cls_logits.to(cpu))
    input_embeddings = th.cat(cls_list, axis=0)  # type: ignore
    cls_logits = th.cat(cls_logit_list, axis=0)  # type: ignore

# Saving Predictions and

torch.save(input_embeddings,
           'bert-finetune_models/{}_embeddings.pt'.format(v.dataset))
torch.save(cls_logits,
           'bert-finetune_models/{}_logits.pt'.format(v.dataset))
# %%
