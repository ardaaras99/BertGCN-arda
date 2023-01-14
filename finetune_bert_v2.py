# %%
from utils_v2 import *
from utils_train_v2 import *
from gcn_models import *
from pathlib import Path
import os
import optuna

WORK_DIR = Path(__file__).parent
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)

docs, y, train_ids, test_ids, NF, FN, NN, FF = load_corpus(v)

# Instead of this if it is 45 you can make it 64(to the next power 2?)
# c_max = max([len(sentence.split()) for sentence in docs])

# if c_max < v_bert.max_length:
#     v_bert.max_length = c_max

nb_train, nb_test, nb_val, nb_class = get_dataset_sizes(
    train_ids, test_ids, y, v.train_val_split_ratio)


# %%
# ADD OPTUNA TYPE HYPERPARAM TUNE HERE

# FINE-TUNE WITH OPTUNA
def objective(trial):
    bert_lr = trial.suggest_float("bert_lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    max_length = trial.suggest_categorical(
        "max_length", [16, 32, 64])  # we cannot fit this for desktop
    #max_length = 16
    bert_init = trial.suggest_categorical(
        'bert_init', ['roberta-base'])

    v_bert.lr = bert_lr
    v_bert.batch_size = batch_size
    v_bert.max_length = max_length
    v_bert.bert_init = bert_init

    print("\n", v_bert)
    model = BertClassifier(
        pretrained_model=v_bert.bert_init, nb_class=nb_class)

    optimizer = th.optim.Adam(model.parameters(), lr=v_bert.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    input_ids_, attention_mask_ = encode_input(
        v_bert.max_length, list(docs), model.tokenizer)

    loader, dataset_sizes, label = configure_bert_inputs(
        input_ids_, attention_mask_, y, nb_train, nb_val, nb_test, v_bert)

    bert_fine_tuner = BertTrainer(v, v_bert, gpu, cpu,
                                  model, optimizer, scheduler, criterion,
                                  loader, dataset_sizes, label)

    bert_fine_tuner.train_val_loop()
    with th.no_grad():
        _, test_w_f1, test_macro, test_micro, test_acc = bert_fine_tuner.loop(
            phase='test')

    print("Test weighted f1 is: {:.3f}".format(100*test_w_f1))
    print("Test macro f1 is: {:.3f}".format(100*test_macro))
    print("Test micro f1 is: {:.3f}".format(100*test_micro))
    print("Test acc is: {:.3f}".format(100*test_acc))

    return test_micro


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # type: ignore
trial = study.best_trial
# %%
print('\n', "*******Re-Train with Best Param Setting*******".center(100), '\n')
v_bert.lr = trial.params['bert_lr']
v_bert.batch_size = trial.params['batch_size']
v_bert.max_length = trial.params['max_length']
v_bert.bert_init = trial.params['bert_init']

model = BertClassifier(
    pretrained_model=v_bert.bert_init, nb_class=nb_class)

optimizer = th.optim.Adam(model.parameters(), lr=v_bert.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
criterion = nn.CrossEntropyLoss()
input_ids_, attention_mask_ = encode_input(
    v_bert.max_length, list(docs), model.tokenizer)

loader, dataset_sizes, label = configure_bert_inputs(
    input_ids_, attention_mask_, y, nb_train, nb_val, nb_test, v_bert)

bert_fine_tuner = BertTrainer(v, v_bert, gpu, cpu,
                              model, optimizer, scheduler, criterion,
                              loader, dataset_sizes, label)

bert_fine_tuner.train_val_loop()
with th.no_grad():
    _, test_w_f1, test_macro, test_micro, test_acc = bert_fine_tuner.loop(
        phase='test')

bert_fine_tuner.generate_embeddings_and_logits()
print("Test weighted f1 is: {:.3f}".format(100*test_w_f1))
print("Test macro f1 is: {:.3f}".format(100*test_macro))
print("Test micro f1 is: {:.3f}".format(100*test_micro))
print("Test acc is: {:.3f}".format(100*test_acc))

# %%
# best setting for r8
# v_bert.lr = 0.00011139922000579552
# v_bert.batch_size = 128
# v_bert.max_length = 16
# v_bert.bert_init = 'roberta-base'
