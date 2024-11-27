import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re

import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import glob
import itertools
from copy import deepcopy
from sklearn.model_selection import KFold

import datetime

from dataloader import *

from model_distribution import *

# %% md
# Train
# %%
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

conv1d_dim1 = 32
conv1d_dim2 = 64
conv1d_dim3 = 128
dense_dim = 256

learning_rate = 0.001
n_epochs = 2000

criterion_distribution = nn.GaussianNLLLoss()
# %%
# 경로 입력 및 아이디 추출

file_path = "./data/total/"
data_path = glob.glob(file_path + '*')
name = []
for file_name in data_path:
    folder_name = os.path.split(file_name)[1][:7]
    name += [folder_name]

id_name = np.unique(name)
# %%
id_name
# %%
test_id = np.array(['IF03014', 'IF00041', 'IM02040', 'IM98049'])
# test_id = np.array(['IF03014', 'IF00041', 'IM02040', 'IM98042'])
# %%
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
from copy import deepcopy
import datetime

# seed 고정
random_seed = 77
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def train_ddp(rank, world_size):
    # 분산 학습 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 데이터셋 준비 및 K-Fold
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    id_name_trnval = np.setdiff1d(id_name, test_id)
    best_MAE_fold = 0

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(id_name_trnval)):
        train_id = id_name_trnval[train_idx]
        valid_id = id_name_trnval[valid_idx]

        print('Train ID : {}\n Valid ID {}'.format(train_id, valid_id))

        train_dataset_R = Gait_Dataset_Salted(file_path, train_id, right=True)
        train_dataset_L = Gait_Dataset_Salted(file_path, train_id, right=False)
        valid_dataset_R = Gait_Dataset_Salted(file_path, valid_id, right=True)
        valid_dataset_L = Gait_Dataset_Salted(file_path, valid_id, right=False)

        train_dataset = torch.utils.data.ConcatDataset([train_dataset_R, train_dataset_L])
        valid_dataset = torch.utils.data.ConcatDataset([valid_dataset_R, valid_dataset_L])

        # DistributedSampler로 데이터 로더 생성
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size,
                                                                        rank=rank, shuffle=True)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=world_size,
                                                                        rank=rank, shuffle=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, sampler=valid_sampler)

        print('Fold {} Dataloader Load Complete'.format(fold + 1))

        # 모델 정의 및 DDP로 감싸기
        model = Encoder(conv1d_dim1, conv1d_dim2, conv1d_dim3, dense_dim).to(rank)
        model = DDP(model, device_ids=[rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Early Stopping을 위한 변수
        best = 10000
        converge_cnt = 0
        best_MAE = 1000
        best_epoch = 0

        # 학습 시작 시간 기록
        start_time = datetime.datetime.now()
        print(f"Training started at: {start_time}")

        # Training loop
        for epoch in range(n_epochs):
            tot_trn_loss = 0.0

            model.train()
            train_sampler.set_epoch(epoch)  # Shuffle each epoch

            for i, data in enumerate(train_loader):
                inputs_acc, inputs_gyr, inputs_prs, stride_length, mu, sigma, folder_id = data
                inputs_acc, inputs_gyr, inputs_prs, stride_length, mu, sigma = inputs_acc.float(), inputs_gyr.float(), inputs_prs.float(), stride_length.float(), mu.float(), sigma.float()
                inputs_acc, inputs_gyr, inputs_prs = inputs_acc.to(rank), inputs_gyr.to(rank), inputs_prs.to(rank)
                mu, sigma = mu.reshape(-1, 1).to(rank), sigma.reshape(-1, 1).to(rank)
                stride_length = stride_length.reshape(-1, 1).to(rank)

                outputs, var = model(inputs_acc, inputs_gyr, inputs_prs)
                loss = criterion_distribution(mu, outputs, var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_trn_loss += loss.item()

            # Evaluation Mode
            model.eval()
            tot_val_loss = 0
            tot_val_MAE = 0

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs_acc, inputs_gyr, inputs_prs, stride_length, mu, sigma, folder_id = data
                    inputs_acc, inputs_gyr, inputs_prs, stride_length, mu, sigma = inputs_acc.float(), inputs_gyr.float(), inputs_prs.float(), stride_length.float(), mu.float(), sigma.float()
                    inputs_acc, inputs_gyr, inputs_prs = inputs_acc.to(rank), inputs_gyr.to(rank), inputs_prs.to(rank)
                    mu, sigma = mu.reshape(-1, 1).to(rank), sigma.reshape(-1, 1).to(rank)
                    stride_length = stride_length.reshape(-1, 1).to(rank)

                    outputs, var = model(inputs_acc, inputs_gyr, inputs_prs)
                    loss = criterion_distribution(mu, outputs, var)
                    tot_val_loss += loss.item()
                    tot_val_MAE += torch.sum(torch.abs(outputs - stride_length)) / len(stride_length)

            trn_loss = tot_trn_loss / len(train_loader)
            val_loss = tot_val_loss / len(val_loader)
            MAE = tot_val_MAE / len(val_loader)

            # Early Stopping
            if val_loss < best:
                best = np.mean(val_loss)
                best_MAE = MAE
                best_epoch = epoch + 1
                if rank == 0:  # Only save model from rank 0 process
                    torch.save(deepcopy(model.state_dict()), f'./model/L2/L2_fold{fold + 1}.pth')
                converge_cnt = 0
            else:
                converge_cnt += 1

            if converge_cnt > 50:
                print(f'Early stopping: Fold {fold + 1}, Epoch {best_epoch}, Valid Loss {best:.3f}, MAE {best_MAE:.3f}')
                best_MAE_fold += best_MAE
                break

        # 학습 종료 시간 및 경과 시간 계산
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Training ended at: {end_time}")
        print(f"Total training time: {elapsed_time}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
