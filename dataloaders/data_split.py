import torch
import pickle
import numpy as np
import sys
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold

sys.path.append('../')
import configparam
np.random.seed(0)

class data_split:
    def __init__(self, dataset, param, shuffle=False, train_idx=None, test_idx=None):

        self.dataset = dataset
        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        if dataset.use_pre_idx == True:
            print('use pre-defined training list')
            with open(param.split_path + "train.txt", "rb") as fp:
                train_list = pickle.load(fp)
            with open(param.split_path + "test.txt", "rb") as fp:
                test_list = pickle.load(fp)
            self.train_sampler = SubsetRandomSampler(train_list)
            self.val_sampler = SubsetRandomSampler(self.val_indices)
            self.test_sampler = SubsetRandomSampler(test_list)

        else:
            if param.dataset_type == 'kfold':
                print('kfold: ratio: %.2f'%param.train_ratio)
                if shuffle:
                    np.random.shuffle(self.indices)
                num_train = int(np.floor(len(self.indices) * param.train_ratio))
                self.train_indices = self.indices[:num_train]
                self.test_indices = self.indices[num_train:]

                '''
                # using Scikit-learn
                self.train_indices = train_idx
                self.test_indices = test_idx
                '''
                print('train num: %d'%(len(self.train_indices)))
                print('test num: %d' % (len(self.test_indices)))
            elif param.dataset_type == 'xsubj':
                return

            self.train_sampler = SubsetRandomSampler(self.train_indices)
            self.val_sampler = SubsetRandomSampler(self.val_indices)
            self.test_sampler = SubsetRandomSampler(self.test_indices)

            with open(param.split_path + "train.txt", "wb") as fp:
                pickle.dump(self.train_indices, fp)
            # with open(param.split_path + "val.txt", "wb") as fp:
            #     pickle.dump(self.val_indices, fp)
            with open(param.split_path + "test.txt", "wb") as fp:
                pickle.dump(self.test_indices, fp)

    def get_split(self, batch_size=50, num_workers=4):

        print('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        print("DONE\n")
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self, batch_size=50, num_workers=4):
        print('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    def get_validation_loader(self, batch_size=50, num_workers=4):
        print('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    def get_test_loader(self, batch_size=50, num_workers=4):
        print('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader