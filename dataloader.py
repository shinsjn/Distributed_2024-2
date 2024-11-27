import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import os 
import glob
import cv2
import itertools

import ast



class Gait_Dataset_Salted(Dataset):
    def __init__(self, file_path, id_name, right=True):
        self.id_name = id_name
        self.file_path = file_path
        self.right = right
        self.inputs_acc, self.inputs_gyr, self.inputs_prs, self.stride_length, self.mu, self.sigma, self.folder_id = get_sensor_salted(file_path, id_name, right=right)
        
    def __len__(self) :
        return len(self.inputs_acc)
    
    def __getitem__(self, idx):
        return self.inputs_acc[idx], self.inputs_gyr[idx], self.inputs_prs[idx], self.stride_length[idx], self.mu[idx], self.sigma[idx], self.folder_id[idx]


def get_sensor_salted(file_path, id_name, right=True):
    data_path = glob.glob(file_path + '*')
    
    inputs_acc = []
    inputs_gyr = []
    inputs_prs = []
    stride_length = []
    mu = []
    sigma = []
    folder_id = []
    stats_data = pd.read_csv('./data/total_mu_sigma.csv')
    for file_name in data_path:
        folder_name = os.path.split(file_name)[1][:7]
        if folder_name in id_name:
            sensor_data = pd.read_csv(file_name, skiprows=2)            
        else:
            continue
            
        record_num = int(os.path.split(file_name)[1].rstrip('.csv')[12:])

        if right==True:
            try:
                index_data = pd.read_csv('./data/index/{}_salted_gaitrite_right_sl.csv'.format(folder_name))
            except:
                index_data = pd.read_csv('./data/index/{}_salted_gaitrite_right_sl.csv'.format(folder_name), encoding='cp949')
        else:
            try:
                index_data = pd.read_csv('./data/index/{}_salted_gaitrite_left_sl.csv'.format(folder_name))
            except:
                index_data = pd.read_csv('./data/index/{}_salted_gaitrite_left_sl.csv'.format(folder_name), encoding='cp949')


        index_data = index_data[index_data['Test Record #']==record_num].reset_index(drop=True)

        try:
            index_data['HS Index']
        except:
            continue

        stride_length.append(list(index_data['Stride Length']))
        mu.append(list(stats_data[stats_data.iloc[:, 0] == folder_name].iloc[:, 1])*len(index_data['Stride Length'])) 
        sigma.append(list(stats_data[stats_data.iloc[:, 0] == folder_name].iloc[:, 2])*len(index_data['Stride Length']))
        folder_id.append(folder_name.split()*len(index_data['Stride Length']))

        if right==True:
            acc = sensor_data.filter(regex="R_ACC")
            gyr = sensor_data.filter(regex="R_GYRO")
            prs = sensor_data.filter(regex="R_value")
        else:
            acc = sensor_data.filter(regex="L_ACC")
            gyr = sensor_data.filter(regex="L_GYRO")
            prs = sensor_data.filter(regex="L_value")   

        acc = (acc / 1000) * 9.8066
        gyr = gyr / 10
        prs = prs.iloc[:, 4:8]

        scaler = MinMaxScaler()
        acc_norm = scaler.fit_transform(acc)
        gyr_norm = scaler.fit_transform(gyr)
        prs_norm = scaler.fit_transform(prs)

        # 가속도와 자이로 센서 값
        for i in range(len(index_data)):
            hs_index = ast.literal_eval(index_data['HS Index'][i])
            inputs_acc.append(np.transpose(cv2.resize(acc_norm[hs_index[0]:hs_index[1]], dsize=(3, 300))))
            inputs_gyr.append(np.transpose(cv2.resize(gyr_norm[hs_index[0]:hs_index[1]], dsize=(3, 300))))
            inputs_prs.append(np.transpose(cv2.resize(prs_norm[hs_index[0]:hs_index[1]], dsize=(4, 300))))
            
    stride_length = np.round(np.array(list(itertools.chain.from_iterable(stride_length))), 3)
    mu = np.round(np.array(list(itertools.chain.from_iterable(mu))), 3)
    sigma = np.round(np.array(list(itertools.chain.from_iterable(sigma))), 3)
    folder_id = np.array(list(itertools.chain.from_iterable(folder_id)))


        
    return inputs_acc, inputs_gyr, inputs_prs, stride_length, mu, sigma, folder_id