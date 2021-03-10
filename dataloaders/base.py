import torchvision
import numpy as np
from torchvision import transforms
from .wrapper import CacheClassLabel
import torch 
from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

class IDSDataset(Dataset):

    def __init__(self,dataroot,transform=None):
        self.path_ds = './data/dataset.npy'
        self.path_lab = './data/labels.npy'
        self.ds = np.load(self.path_ds)

        self.label = np.load(self.path_lab)
        self.whole = np.concatenate((self.ds, self.label), axis=1)
        self.n_samples= len(self.whole)
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        sample = self.whole[index,:-1],int(self.whole[index,-1])
        if self.transform:
            sample=self.transform(sample)
        return sample

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        targets=np.array([targets])
        all_inputs_normalised = (inputs - inputs.min(axis=0))/(inputs.max(axis=0)-inputs.min(axis=0)+1e-5)
        return torch.from_numpy(all_inputs_normalised), torch.from_numpy(targets)

def train_val_dataset(dataset, val_split=0.33):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split,random_state=42)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def IDS (dataroot, train_aug=False):
    print("dataroot: ", dataroot)
    whole = IDSDataset(dataroot,transform=ToTensor())
    df= train_val_dataset(whole)
    train_dataset, val_dataset = df['train'],df['val']
    train_dataset = CacheClassLabel(train_dataset)
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset