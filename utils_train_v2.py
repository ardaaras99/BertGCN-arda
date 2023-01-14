# %%
import torch.nn as nn
from torch.optim import lr_scheduler
from utils_v2 import *
import torch

import torch as th
from transformers import logging as lg
from gcn_models import GCN_Trainer, GCN_type1, GCN_type2, BertClassifier
lg.set_verbosity_error()


def type_trainer_helper1(A_s, nfeat, v, gpu, nb_class, gcn_type):
    if gcn_type == 1:
        print('Type 1 Training')
        gcn_model = GCN_type1(A_s, nfeat, v, gpu, nclass=nb_class)
        criterion = nn.CrossEntropyLoss()
    else:
        print('Type 2 Training')
        cls_logit = torch.load(
            'bert-finetune_models/{}_logits.pt'.format(v.dataset))
        gcn_model = GCN_type2(
            A_s, nfeat, v, gpu, cls_logit.to(gpu), n_class=nb_class)
        criterion = nn.NLLLoss()
    return gcn_model, criterion


def type_trainer(all_paths, label, v,
                 FF, NF, FN, NN,
                 gpu, nb_train, nb_val, nb_test, nb_class,
                 gcn_type=1):

    for path in all_paths:
        v.gcn_path = path
        print('\nCurrent GCN path: {}'.format(v.gcn_path))
        A1, A2, A3, input_type, nfeat = get_path(v, FF, NF, FN, NN)

        if v.gcn_path == "NF-FN-NF":
            v.n_hidden.append(100)
            A_s = (A1.to(gpu), A2.to(gpu), A3.to(gpu))  # type: ignore
        else:
            A_s = (A1.to(gpu), A2.to(gpu), A3)

        input_embeddings = get_input_embeddings(input_type, gpu, A_s, v)

        gcn_model, criterion = type_trainer_helper1(
            A_s, nfeat, v, gpu, nb_class, gcn_type)

        model_path = 'gcn_models/{}_type{}_weights_{}.pt'.format(
            v.dataset, str(gcn_type), v.gcn_path)

        optimizer = th.optim.Adam(gcn_model.parameters(), lr=v.lr)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[30], gamma=0.1)

        gcn_trainer = GCN_Trainer(
            gcn_model, optimizer, scheduler, label,
            input_embeddings.to(gpu),
            nb_train, nb_val, nb_test,
            v, gpu, criterion,
            model_path=model_path)

        gcn_model = gcn_trainer.train_val_loop()
        gcn_model.load_state_dict(torch.load(model_path))
        _, test_w_f1, test_macro, test_micro = gcn_trainer.eval_model(
            phase='test')

        print("Test weighted f1 is: {:.3f}".format(100*test_w_f1))
        print("Test macro f1 is: {:.3f}".format(100*test_macro))
        print("Test micro f1 is: {:.3f}".format(100*test_micro))
        return test_micro
