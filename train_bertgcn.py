# %%
import copy
from pathlib import Path
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, Engine
import torch.utils.data as Data
from utils.utils_train import *
from utils.utils import *
import torch.nn.functional as F
import torch as th
import os


def take_to(x):
    global g_label, g_train, g_val, g_test, g_label_train, bert_output, gcn_input, g_input_ids, g_attention_mask
    g_label = g_label.to(x)
    g_train = g_train.to(x)
    g_val = g_val.to(x)
    g_test = g_test.to(x)
    g_label_train = g_label_train.to(x)
    bert_output = bert_output.to(x)
    gcn_input = gcn_input.to(x)
    g_input_ids = g_input_ids.to(x)
    g_attention_mask = g_attention_mask.to(x)


def update_feature():
    global model, bert_output, g_input_ids, g_attention_mask
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
    bert_output = cls_feat
    # take_to(cpu)
    return bert_output


# Configure
WORK_DIR = Path(__file__).parent
cur_dir = os.path.basename(__file__)
v, ckpt_dir, config, sh, fh, logger, cpu, gpu = configure(WORK_DIR, cur_dir)
#logger.info('checkpoints will be saved in {}'.format(ckpt_dir))


def train_step(engine, batch):
    global model, optimizer, g_input_ids, g_attention_mask, bert_output, gcn_input, g_label_train, g_train
    model.train()
    # A1, A2, A3 = model.gcn.A_s
    # A1 = bert_output.T
    # A2 = bert_output
    # model.gcn.A_s = (A1.to(gpu), A2.to(gpu), A3)
    model = model.to(gpu)
    take_to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g_train[idx].type(th.BoolTensor)
    y_pred = model(g_input_ids,
                   g_attention_mask, bert_output, gcn_input, idx)[train_mask]

    y_true = g_label_train[idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    bert_output.detach_()  # bu detach neymiş bir bak önemli bir abiye benziyor
    gcn_input.detach_()
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
    global gcn_input, model
    print(model.gcn.A_s[0][0][:10])
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, optimizer, g_input_ids, g_attention_mask, bert_output, gcn_input
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        take_to(gpu)

        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g_input_ids,
                       g_attention_mask, bert_output, gcn_input, idx)
        y_true = g_label[idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics = {
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)

acc_sum = 0


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    global acc_sum
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    acc_sum = acc_sum + test_acc
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc*100, train_nll, val_acc*100, val_nll, test_acc*100, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        #logger.info("New checkpoint")
        if v.train_bert_w_gcn == 'no':
            th.save(
                {
                    'gcn': model.gcn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': trainer.state.epoch,
                },
                os.path.join(
                    ckpt_dir, 'checkpoint.pth'
                )
            )
        else:
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


# %%
all_paths = ["FF-NF", "FN-NF", "NN-NN", "NF-NN", "NF-FN-NF"]
all_paths = ["FN-NF"]
for path in all_paths:
    v.gcn_path = path
    if v.gcn_path == "NF-FN-NF":
        v.n_hidden = [200, 100]
    else:
        v.n_hidden = [200]
    print("\nGCN PATH IS:", str(v.gcn_path))
    model, optimizer, A_s, input_type, train_mask, g_label, g_train, g_val, g_test, g_label_train, g_input_ids, g_attention_mask, idx_loader_train, idx_loader_val, idx_loader_test, idx_loader = set_variables(
        v, gpu, config)
    bert_output = update_feature()

    if input_type == "document-matrix input":
        print("We have input matrix: n_doc x 768")
        gcn_input = copy.deepcopy(bert_output)
    else:
        print("We have input matrix: n_vocab x n_vocab")
        gcn_input = th.eye(A_s[0].shape[1]).to(gpu)

    if v.train_bert_w_gcn == "no":
        for param in model.bert_model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    log_training_results.best_val_acc = 0
    trainer.run(idx_loader, max_epochs=v.nb_epochs)
    print("Interpolation constant found to be:", str(model.m))
    print("Average acc is: " + str(acc_sum / v.nb_epochs))
