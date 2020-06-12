# FairSR -- Fairness-aware Sequential Recommendationthrough Multi-task Learning with Preference GraphEmbeddings

![alt text](https://github.com/fairsr/fairsr/blob/master/FairSR_framework.png)





RetaGNN QQ will go to ICDE. Maybe ICDE is ICDM's friend.

How to implement FairSR on toy dataset?
1. git clone 
2. cd fairsr
3. python3 main.py




Requirements
------------

Latest tested combination: Python 3.8.1 + PyTorch 1.4.0 + PyTorch_Geometric 1.4.2.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.



Usages
------

### Flixster, Douban and YahooMusic

To train on Flixster, type:

    python Main.py --data-name flixster --epochs 40 --testing --ensemble

The results will be saved in "results/flixster\_testmode/". The processed enclosing subgraphs will be saved in "data/flixster/testmode/". Change flixster to douban or yahoo\_music to do the same experiments on Douban and YahooMusic datasets, respectively. Delete --testing to evaluate on a validation set to do hyperparameter tuning.
