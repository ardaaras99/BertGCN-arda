# %%
from utils_scripts.utils_v2 import *
from model_scripts.trainers import configure_bert_trainer_inputs, BertTrainer
from model_scripts.objective_funcs import BertObjective
from pathlib import Path

import optuna
from datetime import datetime

WORK_DIR = Path(__file__).parent
cur_dir = os.path.basename(__file__)
v, _, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)

v.cpu = cpu
v.gpu = gpu
v.dataset = "Ohsumed"
set_seed()
v = load_corpus(v)
v = get_dataset_sizes(v, v.train_val_split_ratio)

#%%
# 1.) Hyperparameter search
param_dict = {
    "bert_lr": [1e-5, 1e-3],
    "batch_size": [32, 64, 128],
    "max_length": [32, 48],
    "bert_init": ["roberta-base"],
    "patience": [2],
}
v.nb_epochs = 10
v.print_gap = 1
v.es_verbose = False
v.model_path = ""
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
study = optuna.create_study(study_name=current_time, direction="maximize")
study.optimize(BertObjective(param_dict, v), n_trials=10)
joblib.dump(
    study,
    "results/{}/hyperparam-studies/bert/{}.pkl".format(v.dataset, study.study_name),
)

# Problem: hyperparam studyde son epoch valuesini tria
# %%

# %%
# 2.) Re-train with best-parameter setting
project_path = os.getcwd()
study_path = os.path.join(
    project_path, "results/{}/hyperparam-studies/bert".format(v.dataset)
)
best_study = find_best_study(study_path)

v = set_best_v_bert(v, best_study.best_trial)
v.nb_epochs = 50
v.print_gap = 1
v.patience = 20
trainer = BertTrainer(configure_bert_trainer_inputs(v))

trainer.train_val_loop()
with torch.no_grad():
    _, test_w_f1, test_macro, test_micro, test_acc = trainer.loop(phase="test")

print("Test weighted f1 is: {:.3f}".format(100 * test_w_f1))
print("Test macro f1 is: {:.3f}".format(100 * test_macro))
print("Test micro f1 is: {:.3f}".format(100 * test_micro))
print("Test acc is: {:.3f}".format(100 * test_acc))

# # saving best model( for the name convention we use the best-study name)
# # we can always access the best model parameter from best_study.best_params
# # normaly we expect single weight file to exist in best-bert-model directory, however we come up with file convention after training best models (98.173 R8 for instance)

# test_acc = round(test_acc * 100, 3)
# trainer.save_model(
#     path="results/{}/best-bert-model/{}_{}_weights.pt".format(
#         v.dataset, test_acc, best_study.study_name
#     )
# )

# # we can also generate embeddings and logits here and save to same destination
# cls_logits, input_embeddings = trainer.generate_embeddings_and_logits()
# torch.save(
#     cls_logits,
#     "results/{}/best-bert-model/{}_{}_logtis.pt".format(
#         v.dataset, test_acc, best_study.study_name
#     ),
# )

# torch.save(
#     input_embeddings,
#     "results/{}/best-bert-model/{}_{}_embeddings.pt".format(
#         v.dataset, test_acc, best_study.study_name
#     ),
# )

# %%
# 3.) Re-generating test results
# Same approach like training with best-param settings, instead we just need test loop


# SUPER IMPORTANT: for testing time in BERT, max length of docs are better since tokenization is influenced, for our existing 98.173 model max length must be 256, so we need to set v to best v in our new design architecture, however these best v got set from existing studies. We need to set it manual for existing ones or maybe retrain it but it will take time.

# one possibility is create fake studies to retrieve best max_length

# Following line of codes are sufficient for our new implementation
# project_path = os.getcwd()
# study_path = os.path.join(
#     project_path, "results/{}/hyperparam-studies/bert".format(v.dataset)
# )
# best_study = find_best_study(study_path)

# v = set_best_v_bert(v, best_study.best_trial)
v.batch_size = 64
v.max_length = 200
v.bert_init = "roberta-base"
trainer = BertTrainer(configure_bert_trainer_inputs(v))

model_path = "results/{}/best-bert-model".format(v.dataset)
weight_path_ext = [
    filename for filename in os.listdir(model_path) if filename.endswith("weights.pt")
][0]
w_path = os.path.join(model_path, weight_path_ext)
trainer.load_model_weights(w_path)
trainer.eval_model(phase="test")
#%%
print("Test weighted f1 is: {:.3f}".format(100 * trainer.metrics["test_w_f1"]))
print("Test acc is: {:.3f}".format(100 * trainer.metrics["test_acc"]))
# %%
cls_logits, input_embeddings = trainer.generate_embeddings_and_logits()
# %%
torch.save(
    cls_logits,
    "results/{}/best-bert-model/{}_{}_{}_logits.pt".format(
        v.dataset,
        "{:.2f}".format(100 * trainer.metrics["test_acc"]),
        v.max_length,
        "2023-01-01-00-00-00",
    ),
)

torch.save(
    input_embeddings,
    "results/{}/best-bert-model/{}_{}_{}_embeddings.pt".format(
        v.dataset,
        "{:.2f}".format(100 * trainer.metrics["test_acc"]),
        v.max_length,
        "2023-01-01-00-00-00",
    ),
)

#%%
# not in normal structure, geçmişi düzeltmek için
# torch.save(
#     cls_logits,
#     "results/{}/best-bert-model/{}_{}_logtis.pt".format(
#         v.dataset, "{:.3f}".format(100 * test_acc), "2023-01-01-00-00-00"
#     ),
# )

# torch.save(
#     input_embeddings,
#     "results/{}/best-bert-model/{}_{}_embeddings.pt".format(
#         v.dataset, "{:.3f}".format(100 * test_acc), "2023-01-01-00-00-00"
#     ),
# )


# best setting for r8
# v.lr = 0.00011139922000579552
# v.batch_size = 128
# v.max_length = 256
# v.bert_init = 'roberta-base'
