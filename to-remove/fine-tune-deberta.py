#%%
import torch
from transformers import (
    DebertaConfig,
    AutoTokenizer,
    DebertaForSequenceClassification,
    AutoModel,
)
from utils_scripts.utils_v2 import *
from utils_train_v2 import *
from model_scripts.trainers import *
from pathlib import Path
import os
import optuna

#%%
WORK_DIR = Path(__file__).parent
cur_dir = os.path.basename(__file__)
v, v_bert, cpu, gpu = configure_jsons(WORK_DIR, cur_dir)
set_seed()
docs, y, train_ids, test_ids, NF, FN, NN, FF = load_corpus(v_bert)

c_max = max([len(sentence.split()) for sentence in docs])
print("Max length for corpus {} is {}".format(v_bert.dataset, c_max))

if c_max < v_bert.max_length:
    v_bert.max_length = c_max

nb_train, nb_test, nb_val, nb_class = get_dataset_sizes(
    train_ids, test_ids, y, v.train_val_split_ratio
)

model_name = "microsoft/deberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# config = DebertaConfig()
# config.output_hidden_states = True
# model = AutoModel.from_pretrained(model_name, config=config)
model = DebertaForSequenceClassification.from_pretrained(
    "microsoft/deberta-base", num_labels=2
)
#%%
# loss = model(**inputs, labels=labels).loss

#%%

v_bert.bert_init = model_name
# model = BertClassifier(pretrained_model=v_bert.bert_init, nb_class=nb_class).to(gpu)

input_ids_, attention_mask_ = encode_input(v_bert.max_length, list(docs), tokenizer)

# input_ids_, attention_mask_ = input_ids_.to(gpu), attention_mask_.to(gpu)
# labels = torch.tensor([0, 1]).to(gpu)
i, a = input_ids_[:2], attention_mask_[:2]
#%%
print(model(input_ids=i, attention_mask=a).logits)
#%%
# class deBertClassifier(th.nn.Module):
#     def __init__(self, pretrained_model="microsoft/deberta-base", nb_class=2):
#         super(deBertClassifier, self).__init__()
#         self.nb_class = nb_class
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#         self.bert = DebertaForSequenceClassification.from_pretrained(
#             pretrained_model, num_labels=self.nb_class
#         )

#     def forward(self, input_ids, attention_mask):
#         logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
#         return logits
