import torch
import numpy as np
import random
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append('../')
import configparam

# attack = 'Gaussian noise'
attack = None
eps = 0.1

class physionet_cnn_loader(Dataset):
    def __init__(self, param):

        if param.use_predefined_idx == 1:
            self.use_pre_idx = True
        else:
            self.use_pre_idx = False

        self.data_path = param.data_path
        target_label = param.target_label
        num_sub = param.num_subject
        num_trial = param.num_trial
        max_num_seq = 1000

        self.eeg_data = []
        self.eeg_label = []

        if target_label == '4':
            print('----------------4 classes--------------')
            gt_type = 1
        elif target_label == '5':
            print('----------------5 classes--------------')
            gt_type = 2

        except_list = [2, 3, 5, 7, 9, 11, 13]

        for s in range(num_sub):
            if param.target_subject[0] != 0 and not s+1 in param.target_subject:
                print('%d is not in target subject list'%(s+1))
                continue
            for v in range(num_trial):
                #  we don't know exact length of each trial. So, if the npy file is not exist, skip to next trial.

                if v+1 in except_list:
                    continue
                for t in range(14):
                    eeg_name = self.data_path+'S%03dR%02dT%03d.npy'%(s+1,v+1,t+1)
                    label_name = self.data_path+'S%03dR%02dT%03d_label.txt'%(s+1,v+1,t+1)
                    if not os.path.exists(label_name):
                        print(label_name)
                    f = open(label_name, 'r')
                    val = float(f.read().replace('\n', ''))

                    if gt_type == 1 and val > 3.0:
                        continue
                    if os.path.exists(eeg_name):
                        self.eeg_data.append(eeg_name)
                        self.eeg_label.append(label_name)
                    else:
                        # print('End: %d'%t)
                        break

        self.len = len(self.eeg_data)
        print(self.len)

    def __getitem__(self, index):
        x = np.load(self.eeg_data[index]).astype(np.float32)
        f = open(self.eeg_label[index], 'r')
        val = float(f.read().replace('\n', ''))
        y = np.int64(val)

        f.close()

        x = x.astype(np.float32)

        # Add Gaussian noise
        if attack == 'Gaussian noise':
            for i in range(len(x)):
                x[i] = x[i] + np.random.randn(x[i].shape[0], x[i].shape[1]) * eps
                x[i] = np.clip(x[i], 0, 1)
        return x, y

    def __len__(self):
        return self.len