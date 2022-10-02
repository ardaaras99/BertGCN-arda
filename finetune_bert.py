# %%
from model import BertClassifier
from torch.optim import lr_scheduler
import logging
import shutil
from sklearn.metrics import accuracy_score
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
import torch.utils.data as Data
import torch as th
import torch.nn.functional as F
from utils import *
import os
from types import SimpleNamespace
import json
from transformers import logging as lg

lg.set_verbosity_error()
WORK_DIR = Path(__file__).parent
CONFIG_PATH = Path.joinpath(
    WORK_DIR, "configs/config_file.json")
config = load_config_json(CONFIG_PATH)

v = SimpleNamespace(**config)  # store v in config

# We decide ckpt_dir here
if v.pretrained_bert_ckpt == "":
    ckpt_dir = 'checkpoint/{}_{}'.format(v.bert_init, v.dataset)
    config["pretrained_bert_ckpt"] = ckpt_dir
    # Serializing json
    json_object = json.dumps(config, indent=4)
    already_trained_flag = False
    # Writing to sample.json
    with open("configs/config_file.json", "w") as outfile:
        outfile.write(json_object)
else:
    ckpt_dir = v.pretrained_bert_ckpt
    already_trained_flag = True

# %%
# Create directory and make copy of original file to that director
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

# I guess this lines helps us to save output as log file
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

#logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

'''
    Data Preprocess 
'''
adj, adj_pmi, adj_tfidf, adj_nf, adj_ff, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    v.dataset)

# Get train,test,val sizes
nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# Define model
model = BertClassifier(pretrained_model=v.bert_init, nb_class=nb_class)

if already_trained_flag:
    print("BERT is already pre-trained, we continue on it")
    ckpt = th.load(os.path.join(
        v.pretrained_bert_ckpt, 'checkpoint.pth'
    ), map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

'''
    y -> nb_node array (also includes words inside of it)
    label -> dictionary that stores labels of each dataset 
    text -> num_of_all_examples,1 stored as list of strings
'''
# transform one-hot label to class ID for pytorch computation
y = th.LongTensor((y_train + y_val + y_test).argmax(axis=1))
label = {}
label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]
# load documents and compute input encodings
corpus_file = './data/corpus/'+v.dataset+'_shuffle.txt'
with open(corpus_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=v.max_length,
                      truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask


input_ids, attention_mask = {}, {}

input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

input_ids['train'], input_ids['val'], input_ids['test'] = input_ids_[
    :nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]

attention_mask['train'], attention_mask['val'], attention_mask['test'] = attention_mask_[
    :nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]

# create train/test/val datasets and dataloaders
datasets = {}
loader = {}
for split in ['train', 'val', 'test']:
    datasets[split] = Data.TensorDataset(
        input_ids[split], attention_mask[split], label[split])
    loader[split] = Data.DataLoader(
        datasets[split], batch_size=v.batch_size, shuffle=True)

# %%
# Training

optimizer = th.optim.Adam(model.parameters(), lr=v.bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, optimizer
    model.train()
    model = model.to(gpu)
    optimizer.zero_grad()
    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(th.long)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc


trainer = Engine(train_step)


def test_step(engine, batch):
    global model
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        y_pred = model(input_ids, attention_mask)
        y_true = label
        return y_pred, y_true


evaluator = Engine(test_step)


metrics = {
    'acc': Accuracy(),
    'nll': Loss(th.nn.CrossEntropyLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(loader['train'])
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['val'])
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['test'])
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "\rEpoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        #logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc
    scheduler.step()


log_training_results.best_val_acc = 0
trainer.run(loader['train'], max_epochs=v.nb_epochs)
