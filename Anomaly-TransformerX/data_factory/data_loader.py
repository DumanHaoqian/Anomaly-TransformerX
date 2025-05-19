import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
    #def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load augmented training data and original data
        aug_data = np.load(data_path + "/fusion_train.npy")#"/fusion_train.npy")
        self.scaler.fit(aug_data)
        self.train = self.scaler.transform(aug_data)        # 114d
        self.train_rec = np.load(data_path + "/SMD_train.npy")  # 38d

        # Load test data and labels
        test_aug = np.load(data_path + "/fusion_test.npy")#"/fusion_test.npy")
        self.test = self.scaler.transform(test_aug)
        self.test_rec = np.load(data_path + "/SMD_test.npy")
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

        # Split validation set
        split_idx = int(len(self.train) * 0.8)
        self.val = self.train[split_idx:]      
        self.val_rec = self.train_rec[split_idx:]

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (
                np.float32(self.train[index:index+self.win_size]),  # 114d
                np.float32(self.train_rec[index:index+self.win_size])  # 38d
            )
        elif (self.mode == 'val'):
            return (
                np.float32(self.val[index:index+self.win_size]),
                np.float32(self.val_rec[index:index+self.win_size])
            )
        elif (self.mode == 'test'):
            return (
                np.float32(self.test[index:index+self.win_size]),  
                np.float32(self.test_rec[index:index+self.win_size]),  
                np.float32(self.test_labels[index:index+self.win_size])  
            )
        else:
            index = index // self.step * self.win_size
            return (
                np.float32(self.test[index:index+self.win_size]),
                np.float32(self.test_rec[index:index+self.win_size]),  
                np.float32(self.test_labels[index:index+self.win_size])
            )


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='SMD'):
    dataset = SMDSegLoader(data_path, win_size, step, mode)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
if __name__=="__main__":

    data_path = "/home/haoqian/anomaly/Anomaly-TransformerX/dataset/SMD"  
    batch_size = 32
    win_size = 100
    step = 50

    train_loader = get_loader_segment(data_path, batch_size, win_size, step, mode='train')
    for batch_idx, (input_data, target_data) in enumerate(train_loader):

        
        print(f"Batch {batch_idx + 1}:")
        print("Input Data Shape:", input_data.shape)
        print("Target Data Shape:", target_data.shape)


        if batch_idx >= 2: 
            break

    val_loader = get_loader_segment(data_path, batch_size, win_size, step, mode='val')

    for batch_idx, (input_data, target_data) in enumerate(val_loader):
        print(f"Validation Batch {batch_idx + 1}:")
        print("Input Data Shape:", input_data.shape)
        print("Target Data Shape:", target_data.shape)
        
        if batch_idx >= 2: 
            break