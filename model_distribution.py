import pandas as pd
import numpy as np
import random

import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os 
import glob
import itertools

# seed 고정
random_seed = 77
torch.manual_seed(random_seed) # torch 
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True # cudnn
torch.backends.cudnn.benchmark = False # cudnn
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random

class Encoder(nn.Module):
    def __init__(self, conv1d_dim1, conv1d_dim2, conv1d_dim3, dense_dim):
        super(Encoder, self).__init__()
             
        self.conv1d_acc = nn.Sequential(
            nn.Conv1d(3, conv1d_dim1, 30),
            nn.BatchNorm1d(conv1d_dim1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1d_dim1, conv1d_dim2, 30),
            nn.BatchNorm1d(conv1d_dim2),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1d_dim2, conv1d_dim3, 30),
            nn.BatchNorm1d(conv1d_dim3),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        self.conv1d_gyr = nn.Sequential(
            nn.Conv1d(3, conv1d_dim1, 30),
            nn.BatchNorm1d(conv1d_dim1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1d_dim1, conv1d_dim2, 30),
            nn.BatchNorm1d(conv1d_dim2),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1d_dim2, conv1d_dim3, 30),
            nn.BatchNorm1d(conv1d_dim3),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        self.conv1d_prs = nn.Sequential(
            nn.Conv1d(4, conv1d_dim1, 30),
            nn.BatchNorm1d(conv1d_dim1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1d_dim1, conv1d_dim2, 30),
            nn.BatchNorm1d(conv1d_dim2),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1d_dim2, conv1d_dim3, 30),
            nn.BatchNorm1d(conv1d_dim3),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        self.dense_mean = nn.Sequential(
            nn.Linear(27264*3, dense_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dense_dim, int(dense_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dense_dim/2), 1),
        )
        
        self.dense_var = nn.Sequential(
            nn.Linear(27264*3, dense_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dense_dim, int(dense_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dense_dim/2), 1),
            nn.Softplus()
        )
        

    def forward(self, inputs_acc, inputs_gyr, inputs_prs):
  
        conv1d_output_acc = self.conv1d_acc(inputs_acc)
        conv1d_output_gyr = self.conv1d_gyr(inputs_gyr)
        conv1d_output_prs = self.conv1d_prs(inputs_prs)
        
        conv1d_output_concat = torch.cat([conv1d_output_acc, conv1d_output_gyr, conv1d_output_prs], dim=1)
        
        mean = self.dense_mean(conv1d_output_concat)
        var = self.dense_var(conv1d_output_concat)
        
        return mean, var