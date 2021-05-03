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
import pandas as pd
from torchvision import transforms
from typing import Dict, Iterable, Callable
from os import path
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
import torch.nn as nn 
from torchsummary import summary
warnings.filterwarnings("ignore")

now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")

train_data_x = []
train_data_y = []
test_data_x= []
test_data_y = []

cache_label = './data/train_data_x.pth'

if(path.exists(cache_label)):

    with open('./data/train_data_x.pth','rb') as f:
        train_data_x = pickle.load(f)
    with open('./data/train_data_y.pth','rb') as f:
        train_data_y = pickle.load(f)

    with open('./data/test_data_x.pth','rb') as f:
        test_data_x = pickle.load(f)

    with open('./data/test_data_y.pth','rb') as f:
        test_data_y = pickle.load(f)
    print('Data Loaded!!!')

else:

    data_path = '../data/dataset.npy'
    labels_path = '../data/labels.npy'
    ds = np.load(data_path)
    label = np.load(labels_path)

    whole = np.concatenate((ds, label), axis=1)

    df = pd.DataFrame(whole,columns=[str(i) for i in range(whole.shape[1])])

    y = df.pop(df.columns[-1]).to_frame()-1
    # print(y.iloc[:,-1].value_counts())
    X = (df-df.min())/(df.max()-df.min() + 1e-5)

    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.33)

    train_dict = {}
    train_label_dict = {}
    test_dict = {}
    test_label_dict = {}

    for i in range(y_train.iloc[:,-1].nunique()):
        train_dict["cat"+str(i)] = X_train[y_train.iloc[:,-1] == i]
        # train_label_dict["cat"+str(i)] = y_train[y_train.iloc[:,-1]==i]
        temp = y_train[y_train.iloc[:,-1]==i]
        print(temp.loc[:,'70'].value_counts())
        if i==0:
            temp.loc[:,'70']=0
        else:
            temp.loc[:,'70']=1
        train_label_dict["cat"+str(i)] = temp
        print(temp.loc[:,'70'].value_counts())

    for i in range(y_test.iloc[:,-1].nunique()):
        test_dict["cat"+str(i)] = X_test[y_test.iloc[:,-1] == i]
        
        temp = y_test[y_test.iloc[:,-1]==i]
        print(temp.loc[:,'70'].value_counts())
        if i==0:
            temp.loc[:,'70']=0
        else:
            temp.loc[:,'70']=1
        test_label_dict["cat"+str(i)] = temp
        print(temp.loc[:,'70'].value_counts())

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

    print('Dumped into ./data/')

# # randseq = torch.randperm(num_classes)
# # class_lists = {str(i):randseq[list(range(split_boundaries[i-1],split_boundaries[i]))].tolist() for i in range(1,len(split_boundaries))}

def task_ordering(perm):

    final_train_data_x = []
    final_train_data_y = []
    final_test_data_x = []
    final_test_data_y = []

    for key,values in perm.items():
        temp_train_data_x = torch.Tensor([])
        temp_train_data_y = torch.Tensor([])
        temp_test_data_x = torch.Tensor([])
        temp_test_data_y = torch.Tensor([])

        for value in values:
            temp_train_data_x = torch.cat([temp_train_data_x,train_data_x[value]])
            temp_train_data_y = torch.cat([temp_train_data_y,train_data_y[value]])
            temp_test_data_x = torch.cat([temp_test_data_x,test_data_x[value]])
            temp_test_data_y = torch.cat([temp_test_data_y,test_data_y[value]])
        
        final_train_data_x.append(temp_train_data_x)
        final_train_data_y.append(temp_train_data_y)
        final_test_data_x.append(temp_test_data_x)
        final_test_data_y.append(temp_test_data_y)

    return final_train_data_x,final_train_data_y,final_test_data_x,final_test_data_y


perm = {'1': [2, 4, 8], '2': [5, 9, 0], '3': [12, 1, 7], '4': [3, 14, 13], '5': [10, 11, 6]}

# print('debug: ',train_data_x[0].shape[0])
# print('len:',len(train_data_x),train_data_x[0].shape)
train_data_x=[x.view(x.shape[0],7,10) for x in train_data_x]
test_data_x=[x.view(x.shape[0],7,10) for x in test_data_x]
# print('debug 2: ',train_data_x[0].shape[0])

dataset = task_ordering(perm)

print(dataset[2][0].shape,dataset[2][1].shape,dataset[2][2].shape,dataset[2][3].shape)
# print(type(train_data_x[0]))
# print((train_data_x[0]))

generic_scenario = tensor_scenario(
    train_data_x=dataset[0],
    train_data_y=dataset[1],
    test_data_x=dataset[2],
    test_data_y=dataset[3],
    task_labels=[int(key)-1 for key in perm.keys()]
)

class Conv(nn.Module):
    def __init__(self, num_classes=2, height=7,width=10,
                 hidden_size=100):
        super().__init__()

        self.conv2d = nn.Conv2d(1,14,kernel_size=3)
        
        self.feature1 = nn.Sequential(nn.Linear(14*5*8, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout())
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._height = height
        self._width = width

    def forward(self, x):
        x=x.view(x.size(0),1,self._height,self._width)
        # (128,7,10)
        x = self.conv2d(x)
        x=x.view(-1,14*5*8)
        x = self.feature1(x)
        x = self.classifier(x)
        return x

# MODEL CREATION
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Conv()
model.to(device)
# summary(model,(7,10),128)
# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard

tb_logger = TensorboardLogger(f'./tb_data/{cur_time}_cnn2d_0task2/')

# log to text file
text_logger = TextLogger(open(f'./logs/{cur_time}_cnn2d_0task2.txt', 'w+'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    ExperienceForgetting(),
    cpu_usage_metrics(experience=True),
    StreamConfusionMatrix(num_classes=2, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

cl_strategy = GEM(
    model, optimizer = Adam(model.parameters()),patterns_per_exp=4400,
    criterion = CrossEntropyLoss(), train_mb_size=128, train_epochs=20, eval_mb_size=128,
    evaluator=eval_plugin,device=device)

# TRAINING LOOP
print('Starting experiment...')

results = []
i=1
for experience in generic_scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    
    print('Training completed')
    torch.save(model.state_dict(), './weights/cnn2d_0task2/After_training_Task_{}'.format(i))
    print('Model saved!')
    print('Computing accuracy on the whole test set')
    i+=1
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(generic_scenario.test_stream))
