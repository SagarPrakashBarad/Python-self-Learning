"""
epoch = 1 forward and backward pass of All training samples

batch_size = number of training samples in one forward and backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size=20 -> 100/20 = 5 iterations for 1 epoch

"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        
        self.n_samples = xy.shape[0]
        self.transform = transform
        
    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class ToTensor:
    
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
    
data = WineDataset()
print(type(data.x), type(data.y))

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
print(type(features), type(labels))