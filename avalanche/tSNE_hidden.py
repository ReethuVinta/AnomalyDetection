import torchvision
import pickle
import numpy as np
import pandas as pd
from torchvision import transforms
from typing import Dict, Iterable, Callable
from os import path
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
import warnings
import random
import os
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")

now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SimpleMLP(nn.Module):

    def __init__(self, num_classes=2, input_size=70, hidden_size=100):
        super(SimpleMLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        inputs = np.array(inputs)
        targets = np.array([targets])
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class IDSDataset(Dataset):
    def __init__(self):
        self.path_ds = './tSNEfigs/stratify/X_t.npy'
        self.path_lab = './tSNEfigs/stratify/y_t.npy'
        self.ds = np.load(self.path_ds)  # each data point of shape (70,)
        self.label = np.load(self.path_lab) 
        self.n_samples = len(self.ds)
        self.transform = ToTensor()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.ds[index], int(self.label[index, -1])
        if self.transform:
            sample = self.transform(sample)
        return sample


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



test_ds = IDSDataset()
test_ds = DataLoader(test_ds, batch_size=64, num_workers=4, worker_init_fn=seed_worker)

# Loading model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading splitted test samples
y_t = np.load('./tSNEfigs/stratify/y_t.npy')

directory='./weights'
cnt=0
for file in os.listdir(directory):  # After each task
    cnt+=1
    model = SimpleMLP(num_classes=2, input_size=70,hidden_size=100)      # Model architecture
    model.eval()  # Eval mode
    model.load_state_dict(torch.load(os.path.join(directory, file), map_location=device))
    model.to(device)
    model.features[2].register_forward_hook(get_activation('features[2]')) 
    out = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_ds):
            data = data.to(device)     # Converting to cuda tensors
            output = model(data)
            act = activation['features[2]']
            out.append(act)
        whole_x = [t.cpu().detach().numpy() for t in out]
        whole_x = np.vstack(whole_x)
        whole_x = np.array(whole_x)
        tsne = TSNE(n_components=2, random_state=0,
                    verbose=1, perplexity=50, init='pca')            
        X_2d = tsne.fit_transform(whole_x)
        np.save('./tSNEfigs/X_2dvalues/'+str(cnt)+'.npy',X_2d)
        target_ids = [0, 1]  # (data_bin_hot['label'].nunique())
        plt.figure(figsize=(10, 6))
        colors = 'r', 'g'
        target_names = ['Normal', 'Attack']
        for i, c, label in zip(target_ids, colors, target_names):
            plt.scatter(X_2d[y_t[:, -1] == i, 0],
                        X_2d[y_t[:, -1] == i, 1], c=c, label=label)
        plt.legend()
        plt.savefig("./tSNEfigs/graphs/task_"+str(cnt)+'.png')
