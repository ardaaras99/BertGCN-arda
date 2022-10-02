# How to Run
This document summarizes the possible configurations that could be use to train BertHete-GCN

## Changing Configurations
One can change the configurations of model by editing config_file.json.

## Fine-Tuning BERT
This section is optional. Original BERT model is already pre-trained. However, we can also fine-tune it on our dataset.

## Training GCN 
While training GCN we can either train it with BERT simultaneously or just train the GCN part. This can be achieved with the `train-bert_w_gcn` parameter. Further, we can choose among different path possibilities to configure our network. This is achieved by `gcn_path` parameter.

## Run Procedure
We will run single file deneme_ignite.py where we can accomplish everything we want.

- Build graphs for desired dataset, note that we always create from 0, not use existing one. Do something for that 
- Decide whether you want to fine_tune BERT or not by `fine_tune_bert` parameter
- Decide which GCN path you want to train
- Decide whether you want to train bert with GCN by `train_bert_w_gcn` parameter
- User must give 3hidden dimensions if X-TX-X path is used, otherwise they should provide 2. In general number of graphs used in path - 1, is equal to amount of different hidden dimensions to be used.

# Training Procedure

- First finetune BERT on data, then do not train with the model to fasten procedure.

# To - Do List

- [x] finetune berti kullanmıyor tunaya sor (pathi veriyorum fakat erişemiyor
- [x] yalnızca bir path implement edilmiş durumda diğer pathleri de configure edebilmek lazım güzel şekilde
- [x] NN graphı için sklearn k-neigh graph bak
- [ ] adamların yaptığı garip train-mask olayı nasıl trainde var testte yok tunaya göster
- [ ] adamların bert parameterlerini kullanabilirsin
- [ ] cehckpointe save ediyor evet ama onu kullanmıyor bir dahaki runda
- [ ] 0.4 m le runlamak 89.87 acc verdi bir epoch
- [x] outputu concatlama fikri de güzel olabilir (bunları mod yapabilirsin config içine koyarsın)
- [ ] ensemble modeller için biraz fikir edinebilirsin
- [ ] data augmentation yapmak hile olur mu? (muhtemelen)
- [ ] Here(ALL PATH TRAIN LOOP) we create everything froms scratch, this can be partioned to just change
- [ ] son pathte hidden dim az olmasına rağmen çalıştı bak
- [ ] pathleri üst üste runlayınca bertgcn checkpointini aynı yere save ediyor, onu kullanmasak bile yanlış approach

