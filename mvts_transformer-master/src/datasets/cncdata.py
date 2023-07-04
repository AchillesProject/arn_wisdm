import os
import tensorflow as tf
import tensorflow_datasets as tfds
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


cuda=True
batch_size=6

class CNCDataset(Dataset):
    def __init__(self, file_name):
        tf_dataset = tf.data.Dataset.load(path=file_name)
        self.list_data = list(tf_dataset)
        
    def __len__(self):
        return len(self.list_data)
    
    def __getitem__(self, idx):
        item = self.list_data[idx]
        x = np.asarray(item[0], dtype='double').copy()
        #x.setflags(write=1)
        y = np.asarray(item[1], dtype='double').copy()
        #y.setflags(write=1)
        #print("numpy flags", x.flags, y.flags)
        return torch.tensor(x), torch.tensor(y), idx

