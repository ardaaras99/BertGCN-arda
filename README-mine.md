# How to Run
This document summarizes the possible configurations that could be use to train BertHete-GCN

## Changing Configurations
One can change the configurations of model by editing config_file.json.

## Fine-Tuning BERT
This section is optional. Original BERT model is already pre-trained. However, we can also fine-tune it on our dataset.

## Training GCN 
While training GCN we can either train it with BERT simultaneously or just train the GCN part. This can be achieved with the `train-bert_w_gcn` parameter. Further, we can choose among different path possibilities to configure our network. This is achieved by `gcn_path` parameter.

'''
    We have multiple possible paths to train listed as follows:
    
    we used the following convention to represent matrices in code:
        FF for F
        NF for X
        FN for TX
        NN for N

    Paper Path Names

        F -> X 
        X -> N 
        TX -> X 
        NN -> NN


    Code Path Names
    
        FF -> NF (for this path we also need to change input which is g_cls_feats) give I
        NF -> NN (for this path we also need to change input which is g_cls_feats) give I
        FN -> NF (this one works no error)
        NN -> NN (this one works no error)
'''

## Run Procedure
We will run single file deneme_ignite.py where we can accomplish everything we want.

- Build graphs for desired dataset, note that we always create from 0, not use existing one. Do something for that 
- Decide whether you want to fine_tune BERT or not by `fine_tune_bert` parameter
- Decide which GCN path you want to train
- Decide whether you want to train bert with GCN by `train_bert_w_gcn` parameter
- User must give 3hidden dimensions if X-TX-X path is used, otherwise they should provide 2. In general number of graphs used in path - 1, is equal to amount of different hidden dimensions to be used.