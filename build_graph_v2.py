# %%
from sklearn.neighbors import kneighbors_graph
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
import nltk
from types import SimpleNamespace
from pathlib import Path
from utils_scripts.utils_v2 import *
import torch

# %%
set_seed()
CONFIG_PATH = "/Users/ardaaras/Desktop/projects/BertGCN-arda/configs/config_file.json"
config = load_config_json(CONFIG_PATH)

v = SimpleNamespace(**config)  # store v in config

# %%

doc_name_list, train_ids, test_ids = [], [], []
y = []
f = open("data/" + v.dataset + ".txt", "r")
lines = f.readlines()

for i, line in enumerate(lines):
    doc_name_list.append(line.strip())
    y.append(line.strip().split("\t")[2])
    temp = line.split("\t")
    if temp[1].find("test") != -1:
        test_ids.append(i)
    elif temp[1].find("train") != -1:
        train_ids.append(i)
f.close()

doc_content_list = []

f = open("data/corpus/" + v.dataset + ".clean.txt", "r")
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()

# Preprocessing Documents by NLTK
if v.pre_process_text == "True":
    print("We do pre processing")
    nltk.download("stopwords")
    nltk.download("punkt")
    stop_words = stopwords.words("english")
    porter = PorterStemmer()
    for i, doc in enumerate(doc_content_list):
        doc = doc.lower()
        doc_p_removed = "".join(
            [char for char in doc if char not in string.punctuation]
        )
        doc_filtered = [
            word for word in word_tokenize(doc_p_removed) if word not in stop_words
        ]
        doc_stemmed = [porter.stem(word) for word in doc_filtered]
        doc_pp = " ".join(doc_filtered)
        doc_content_list[i] = doc_pp
else:
    print("No PreProcess")

y = np.array(y)
doc_content_list = np.array(doc_content_list)

np.random.shuffle(train_ids)
np.random.shuffle(test_ids)

ids = train_ids + test_ids
doc_content_list = doc_content_list[ids]
y = y[ids]

print("Total number of documents: " + str(len(doc_content_list)))
print("Number of initial training documents: " + str(len(train_ids)))
print("Number of initial test documents: " + str(len(test_ids)))

# %%

# Building Vocab

word_freq = {}
word_set = set()
for docs in doc_content_list:
    words = docs.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab, vocab_size = list(word_set), len(list(word_set))

word_doc_list = {}  # id: word key: wordün geçtiği dökümanların idsi

for i, doc in enumerate(doc_content_list):
    words = doc.split()
    appeared = set()
    for word in words:
        if word in appeared:
            # skip that iteration and go to next for loop element
            continue
        if word in word_doc_list:
            # create temp doc_list of existing document list
            tmp = word_doc_list[word]
            tmp.append(i)
            word_doc_list[word] = tmp
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_id_map = {}

for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# %%

# PPMI
window_size, windows = 20, []

# Create all windows
for doc_words in doc_content_list:
    words = doc_words.split()
    l = len(words)
    if l <= window_size:
        windows.append(words)
    else:
        for j in range(l - window_size + 1):
            window = words[j : j + window_size]
            windows.append(window)

word_window_freq = {}  # w(i) ids: original words
for window in windows:
    appeared = set()
    for i, word in enumerate(window):
        if word in appeared:
            continue
        if word in word_window_freq:
            word_window_freq[word] += 1
        else:
            word_window_freq[word] = 1
        appeared.add(word)

word_pair_count = {}  # w(i,j) ids: word ids (can be looked from word_id_map)
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]

            if word_i_id == word_j_id:
                continue

            word_pair_str = str(word_i_id) + "," + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + "," + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row_ff, col_ff, weight_ff = [], [], []
num_window = len(windows)

for key in word_pair_count:
    temp = key.split(",")
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log(
        (1.0 * count / num_window)
        / (1.0 * word_freq_i * word_freq_j / (num_window * num_window))
    )
    if pmi <= 0:
        continue

    row_ff.append(i)
    col_ff.append(j)
    weight_ff.append(pmi)

# %%
# TFIDF

doc_word_freq = {}

"""
    calculating term frequency
"""
for doc_id, doc_words in enumerate(doc_content_list):
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + "," + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1


"""
    calculating inverse document frequency
"""
row_nf, col_nf, weight_nf = [], [], []

for i, docs in enumerate(doc_content_list):
    words = docs.split()
    doc_word_set = set()
    for word in words:
        k = 0  # to track test documents
        train_flag, test_flag = False, False
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + "," + str(j)
        freq = doc_word_freq[key]

        row_nf.append(i)
        col_nf.append(j)
        idf = log(1.0 * len(doc_content_list) / word_freq[vocab[j]])

        weight_nf.append(freq * idf + 1e-6)
        doc_word_set.add(word)

# %%
# OBTAIN GRAPHS
NF = sp.csr_matrix(
    (weight_nf, (row_nf, col_nf)), shape=(len(doc_content_list), vocab_size)
)

FF = sp.csr_matrix((weight_ff, (row_ff, col_ff)), shape=(vocab_size, vocab_size))

n_neighbors = 25
metric = "cosine"
NN = kneighbors_graph(NF, n_neighbors, metric=metric, include_self=True)

# max normalization
FN = NF.T


def max_min_normalize(x):
    x_normed = (x - x.min(0, keepdim=True)[0]) / (
        x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0]
    )
    return x_normed


# %%
if v.normalize_graphs == "True":
    print("Normalization of graphs")
    NN, NF, FN, FF = (
        to_torch_sparse_tensor(NN),
        to_torch_sparse_tensor(NF),
        to_torch_sparse_tensor(FN),
        to_torch_sparse_tensor(FF),
    )

    NF = sp.csr_matrix(max_min_normalize(NF.to_dense()))
    FN = NF.T
    FF = sp.csr_matrix(max_min_normalize(FF.to_dense()))
    NN = sp.csr_matrix(max_min_normalize(NN.to_dense()))
print("No normalization of graphs")
NN, NF, FN, FF = (
    to_torch_sparse_tensor(NN),
    to_torch_sparse_tensor(NF),
    to_torch_sparse_tensor(FN),
    to_torch_sparse_tensor(FF),
)

# %%
# Save Graphs
# datas/dataset_name/graphs
torch.save(NF, "datas/{}/graphs/{}_NF.pt".format(v.dataset, v.dataset))
torch.save(FN, "datas/{}/graphs/{}_FN.pt".format(v.dataset, v.dataset))
torch.save(NN, "datas/{}/graphs/{}_NN.pt".format(v.dataset, v.dataset))
torch.save(FF, "datas/{}/graphs/{}_FF.pt".format(v.dataset, v.dataset))

# %%
# Converting Labels to one-hot vectors

label_set = set(y)
label_list = list(label_set)
y_one_hot = []
for label in y:
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y_one_hot.append(one_hot)

y = np.array(y_one_hot)
train_ids = np.array(train_ids)
test_ids = np.array(test_ids)


def dump_obj(obj, obj_name, v):

    f = open("datas/{}/{}_{}".format(v.dataset, v.dataset, obj_name), "wb")
    pkl.dump(obj, f)
    f.close()


# docs,y,train_ids,test_ids
objs = [
    (doc_content_list, "docs"),
    (y, "y"),
    (train_ids, "train_ids"),
    (test_ids, "test_ids"),
]

for obj_tuple in objs:
    obj, obj_name = obj_tuple
    dump_obj(obj, obj_name, v)

# %%
