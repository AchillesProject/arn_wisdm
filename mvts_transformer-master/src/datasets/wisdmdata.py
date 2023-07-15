import os, sys
import tensorflow as tf
import tensorflow_datasets as tfds
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class WISDMDataset(Dataset):
    def __init__(self, file_path, noIn=3, noOut=6):      
        self.data = np.array(pd.read_csv(file_path, skiprows=1))
        with open(file_path, "r") as fp:
            [self.noIn, self.noOut] = [int(x) for x in fp.readline().replace('\n', '').split(',')]
        self.timestep = self.data.shape[1]/(self.noIn+self.noOut)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx): # X.shape(batches, 
        x, y = None, None
        print(self.data.shape, idx)
        for i in range(self.data.shape[0]):
            x_i = self.data[i].reshape(-1, self.noIn + self.noOut)
            x_i, y_i = x_i[:, 0:self.noIn], x_i[:, self.noIn:]
            # x = x_i[np.newaxis,:,:] if x is None else np.append(x, x_i[np.newaxis,:,:], axis=0)
            x = x_i if x is None else np.append(x, x_i, axis=0)
            y = y_i if y is None else np.append(y, y_i, axis=0)
        print(x.shape, y.shape, idx)
        sys.exit()
        return torch.from_numpy(x), torch.from_numpy(y), idx
