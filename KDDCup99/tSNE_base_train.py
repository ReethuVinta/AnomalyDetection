from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import ExperienceForgetting, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import GEM
import torchvision
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from torchvision import transforms
from typing import Dict, Iterable, Callable
import os 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import avalanche
from avalanche.benchmarks.generators import tensor_scenario
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")

train_data_x = []
train_data_y = []
test_data_x= []
test_data_y = []

cache_label = './data/train_data_x.pth'

if(os.path.exists(cache_label)):

    with open('./data/train_data_x.pth','rb') as f:
        train_data_x = pickle.load(f)
    with open('./data/train_data_y.pth','rb') as f:
        train_data_y = pickle.load(f)

    with open('./data/test_data_x.pth','rb') as f:
        test_data_x = pickle.load(f)

    with open('./data/test_data_y.pth','rb') as f:
        test_data_y = pickle.load(f)

else:
    with open('./data/kdd_df.pth','rb') as f:
        df = pickle.load(f)

    y = df.pop(df.columns[-1]).to_frame()
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.33)

    train_dict = {}
    train_label_dict = {}
    test_dict = {}
    test_label_dict = {}

    for i in range(y_train.iloc[:,-1].nunique()):
        train_dict["cat"+str(i)] = X_train[y_train.iloc[:,-1] == i]
        temp = y_train[y_train.iloc[:,-1] == i]
       
        if i==1:
            temp.iloc[:,-1]=0
        else:
            temp.iloc[:,-1]=1
        train_label_dict["cat"+str(i)] = temp

    for i in range(y_test.iloc[:,-1].nunique()):
        test_dict["cat"+str(i)] = X_test[y_test.iloc[:,-1] == i]
        temp = y_test[y_test.iloc[:,-1]==i]

        if i==1:
            temp.iloc[:,-1]=0
        else:
            temp.iloc[:,-1]=1
        test_label_dict["cat"+str(i)] = temp

    train_data_x = list(torch.Tensor(train_dict[key].to_numpy()) for key in train_dict)
    train_data_y = list(torch.Tensor(train_label_dict[key].to_numpy()) for key in train_label_dict)
    test_data_x = list(torch.Tensor(test_dict[key].to_numpy()) for key in test_dict)
    test_data_y = list(torch.Tensor(test_label_dict[key].to_numpy()) for key in test_label_dict)


    with open('./data/train_data_x.pth','wb') as f:
        pickle.dump(train_data_x,f)

    with open('./data/train_data_y.pth','wb') as f:
        pickle.dump(train_data_y,f)

    with open('./data/test_data_x.pth','wb') as f:
        pickle.dump(test_data_x,f)

    with open('./data/test_data_y.pth','wb') as f:
        pickle.dump(test_data_y,f)



whole_x = [t.numpy() for t in train_data_x]
whole_x = np.vstack(whole_x)
whole_x = np.array(whole_x)

whole_y = [t.numpy() for t in train_data_y]
whole_y = np.vstack(whole_y)
whole_y = np.array(whole_y)

# Splitting test samples
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.95, random_state=0)
sss.get_n_splits(whole_x, whole_y)
for train_index, test_index in sss.split(whole_x, whole_y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_t, X_te = whole_x[train_index], whole_x[test_index]
    y_t, y_te = whole_y[train_index], whole_y[test_index]
    break

tsne = TSNE(n_components=2, random_state=0,verbose=1,perplexity=50,init='pca')
X_2d = tsne.fit_transform(X_t)

target_ids = [0,1] 
y_t=y_t.astype('int')

plt.figure(figsize=(10,6))
colors = 'r', 'g'
target_names=['Normal','Attack']
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[y_t[:,-1] == i,0], X_2d[y_t[:,-1] == i,1], c=c, label=label)
plt.legend()

plt.savefig("./tsne_kdd_train.png")
