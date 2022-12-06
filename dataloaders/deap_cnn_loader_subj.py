import torch
import numpy as np
import random
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append('../')
import configparam

#attack = 'Gaussian noise'
attack = None
eps = 0.1

class deap_cnn_loader(Dataset):
    def __init__(self, param, subject_list=None):

        if param.use_predefined_idx == 1:
            self.use_pre_idx = True
        else:
            self.use_pre_idx = False

        self.data_path = param.data_path
        self.subject_list = subject_list
        target_label = param.target_label
        num_sub = param.num_subject
        num_trial = param.num_trial
        max_num_seq = 1000

        self.eeg_data = []
        self.eeg_label = []
        self.gt_type = 0

        if target_label == 'valence':
            print('----------------valence--------------')
            self.gt_type = 1
        elif target_label == 'arousal':
            print('----------------arousal--------------')
            self.gt_type = 2
        elif target_label == 'both':
            print('----------------both--------------')
            self.gt_type = 3

        for s in self.subject_list:
            if param.target_subject[0] != 0 and not s+1 in param.target_subject:
                print('%d is not in target subject list'%(s+1))
                continue
            for v in range(num_trial):
                #  we don't know exact length of each trial. So, if the npy file is not exist, skip to next trial.
                for t in range(10, max_num_seq):
                    eeg_name = self.data_path+'S%02dT%02d_%04d.npy'%(s+1,v+1,t+1)
                    if os.path.exists(eeg_name):
                        self.eeg_data.append(self.data_path + 'S%02dT%02d_%04d.npy'%(s+1,v+1,t+1))
                        if self.gt_type == 1:
                            self.eeg_label.append(self.data_path + 'S%02dT%02d_%04d_valence.txt'%(s+1,v+1,t+1))
                        if self.gt_type == 2:
                            self.eeg_label.append(self.data_path + 'S%02dT%02d_%04d_arousal.txt'%(s+1,v+1,t+1))
                        if self.gt_type == 3:
                            self.eeg_label.append(self.data_path + 'S%02dT%02d_%04d_multi.txt'%(s+1,v+1,t+1))
                    else:
                        # print('End: %d'%t)
                        break

        self.len = len(self.eeg_data)
        print(self.len)

    def __getitem__(self, index):
        x = np.load(self.eeg_data[index]).astype(np.float32)
        f = open(self.eeg_label[index], 'r')
        val = float(f.read().replace('\n', ''))
        if self.gt_type < 3:
            if val >= 5:
                y = 1
            else:
                y = 0
        else:
            y = int(val)

        f.close()

        x = x.reshape(-1, x.shape[0], x.shape[1])
        x = x.astype(np.float32)

        # Add Gaussian noise
        if attack == 'Gaussian noise':
            for i in range(len(x)):
                x[i] = x[i] + np.random.randn(x[i].shape[0], x[i].shape[1]) * eps
                x[i] = np.clip(x[i], 0, 1)
        return x, y

    def __len__(self):
        return self.len