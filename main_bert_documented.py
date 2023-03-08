# %%
from utils_scripts.utils_v2 import *
from utils_scripts.utils_train_v2 import set_best_v_bert
from model_scripts.trainers import configure_bert_trainer_inputs, BertTrainer
from model_scripts.objective_funcs import BertObjective
from pathlib import Path

import optuna
from datetime import datetime

#%%
WORK_DIR = Path(__file__).parent
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)

set_seed()

docs, y, train_ids, test_ids, NF, FN, NN, FF = load_corpus(v_bert)
nb_train, nb_test, nb_val, nb_class = get_dataset_sizes(
    train_ids, test_ids, y, v_bert.train_val_split_ratio
)

# Save all intermediate parameters as an instance of v_bert object
v_bert.v = v
v_bert.cpu = cpu
v_bert.gpu = gpu
v_bert.docs = docs
v_bert.y = y
v_bert.train_ids = train_ids
v_bert.test_ids = test_ids
v_bert.nb_train = nb_train
v_bert.nb_test = nb_test
v_bert.nb_val = nb_val
v_bert.nb_class = nb_class

#%%

# Comments on following block of code: We can find the maximum length in a document with following lines, however choosing max length of tokenizer as the max length of sentence is not optimal. Powers of 2 works better.
# c_max = max([len(sentence.split()) for sentence in docs])
# print("Max length for corpus {} is {}".format(v_bert.dataset, c_max))

# if c_max < v_bert.max_length:
#     v_bert.max_length = c_max


"""
Welcome to the playground of BERT module, here we allow you to do several different 
things and analyze the models performance.
    1.) You can do hyperparameter search on BERT or RoBERTa with simple configuration on param dict.
    2.) You can re-train BERT variant with the best hyperparameter setting you found previously. 
    3.) Check the existing BERT models and their accuracies, then run the model the obtain inference performance. It is also a sanity check whether we saved models correctly or not.
    4.) You can save the trained models with their embeddings, weights and logits.
    5.) You can train BERT model with desired parameter settings.

Below we will show demonstration for each and every possible settings
"""

# 1.) Hyperparameter search on BERT variants,
# Suggestion: One suggestion is you can have smaller patience for hyperparam search, then you can train best resultant hyperparam setting for longer duration.

# Missing: saving the trial with name, to somewhere possibly for later access.

"""
param_dict:
    bert_lr -> 1st value is the min val for lr, and 2nd is the max. We sweep lr in log scale.
    Rest of the parameters are categorical, you can simply add other values to try as a value to list.
    batch_size -> batch size of the model
    max_length -> maximum possible length for the BERT tokenizer
    bert_init -> model name of the BERT variant downloaded from hugging face
    patience -> patience for the early stop mechanism
"""
param_dict = {
    "bert_lr": [5e-5, 1e-4],
    "batch_size": [16, 64],
    "max_length": [128, 256],
    "bert_init": ["roberta-base"],
    "patience": [1],
}

"""
v_bert -> is our data class object, we access necessary parameters from it
n_trials -> number of different trials to get 
best_trial -> best trial of the study after hyperparameter optimization
best_params -> best parameter configuration
"""

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
study = optuna.create_study(study_name=current_time, direction="maximize")
study.optimize(BertObjective(param_dict, v_bert), n_trials=1)
joblib.dump(
    study,
    "results/{}/hyperparam-studies/bert/{}.pkl".format(
        v_bert.dataset, study.study_name
    ),
)


# %%
# 2.) Re-train with best-parameter setting
# Suggestion: As stated before, we can train best param setting for longer duration


# TO-DO:
#   2) Since all possiblities we provide for users in single file and we run top the bottom, the top cell configuration is same for everyone and must be run all the time. Maybe we can come with better approach


# we need to find best_study first to reach best trial
project_path = os.getcwd()
study_path = os.path.join(
    project_path, "results/{}/hyperparam-studies/bert".format(v_bert.dataset)
)


#%%

best_study = find_best_study(study_path)
v_bert = set_best_v_bert(v_bert, best_study.best_trial)
v_bert.patience = 20
bert_fine_tuner = BertTrainer(*configure_bert_trainer_inputs(v_bert))

bert_fine_tuner.train_val_loop()
with th.no_grad():
    _, test_w_f1, test_macro, test_micro, test_acc = bert_fine_tuner.loop(phase="test")

print("Test weighted f1 is: {:.3f}".format(100 * test_w_f1))
print("Test macro f1 is: {:.3f}".format(100 * test_macro))
print("Test micro f1 is: {:.3f}".format(100 * test_micro))
print("Test acc is: {:.3f}".format(100 * test_acc))
# %%
bert_fine_tuner.generate_embeddings_and_logits()

# 3.) You can always access the model from the saved models.

# Reminder: - We save the model weights at the end of train_val_loop() function
#           - We save embeddings and logits with, generate_embeddings_and_logits() function

# TO-DO: we need better path naming convention for BERT model. One possibility is to use accuracy, but we also need to save best_trial with same convention or maybe whole study. Also we need to avoid saving if better model exists


def get_trainer(path):

    """
    For testing model, we need the weights of the model to load, then we can run with test loop, but we can achive test loop with a BertTrainer, thus we need to load model first then pass it to bert trainer
    """
    pass


# best setting for r8
# v_bert.lr = 0.00011139922000579552
# v_bert.batch_size = 128
# v_bert.max_length = 16
# v_bert.bert_init = 'roberta-base'
