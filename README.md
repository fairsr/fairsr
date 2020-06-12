# FairSR -- Fairness-aware Sequential Recommendationthrough Multi-task Learning with Preference GraphEmbeddings

![alt text](https://github.com/fairsr/fairsr/blob/master/FairSR_framework.png)








Requirements
------------

Latest tested combination: Python 3.6.6 + PyTorch 1.4.0.

Install [PyTorch](https://pytorch.org/)

Other required python libraries:numpy,tqdm,sklearn,pickle etc.


Usages
------

### Instagram Check-in Toy Dataset
Train and evaluate the model (you are strongly recommended to run the program on a machine with GPU)

    python main.py 

The results will show the performance of FairSR by ranking metric Precision@k, NDCG@k and Recall@k.
