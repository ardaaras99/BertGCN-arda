# Current Weakness of the Code

## 1. Testing the Learned Weights

1. When model is trained and saved, we need a script to run(only the forward pass) quick test and validate the obtained results. It will help.
2. We need to run model with best parameter settings, that is find during hyperparameter optimiziation.
3. We save models on top of the current directory, sometimes it is useful, but for better approach, we need to same with time stamping and the obtained test-accuracy results.
   1. For this, saved the current approaches with time stamp and test accuracy value.
4. Generate any existing result without an effor.
   
## 2. Timing Comparasion

1. Currently, we cannot run the other models, find solutions for that.

## 3. Hyperparameter Tuning

1. Although we can tune, we must use *args and **kwargs to call hyperparam tune from another function. Generalize this to all models as much as possible.

## 4. Lots of Python Scripts

1. Currently we have lots of python scripts which might be unnecassary, also some of the functions make project harder to read. Fix these issues.

## 5. Type IV problem in Mac

1. We cannot run Type IV in our pc since some of the libraries(sparse library as far as I remember), is not implemented with mps support. Check this issue again.

## 6. DeBERTa

1. I tried to train DeBERTa but cannot reach the results mentioned in papers. Also I cannot retrieve embeddings as in RoBERTa and BERT. Possibly remove the file.


## v & v_bert conflict

1. Still not decided how to overcome this. Two configuration jsons seems unnecessary.

## Current Problems

1. We save at earlystopping poitns, to save model, we need to reach BertTrainer.model to save which is easy to reach. We can do it in another phase no need to save at the end of train_val loop.
2. Hyper-param search ederken model biraz trainleniyor daha az epoch ve patiencela, buradan en iyi parametreleri seçiyoruz ve sıfırdan tekrar daha uzun trainleyelim diyoruz. Bunun yerine en iyi olanı save edip onun kaldığı yerden devam edebiliriz. 