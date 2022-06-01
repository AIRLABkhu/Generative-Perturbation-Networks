import numpy as np
import os
import pandas as pd


# from make_dataset
# num_subject   = 20;
# num_trial     = 84;
# num_channel   = 64;
# num_data      = 640;
# Time_consider = 4 / 10;
# Data_points   = Time_consider * 160;

# first I saved all csv files without shuffle
# - disable: rowrank in make_dataset.m
#  So, All dataset is ordered

eeg_name = '/home/hyoseok/research/eeg/code/EEG-DL-master/Preprocess_EEG_Data/For-CNN-based-Models/all_data.csv'
label_name = '/home/hyoseok/research/eeg/code/EEG-DL-master/Preprocess_EEG_Data/For-CNN-based-Models/all_labels.csv'

eegs = pd.read_csv(eeg_name, sep=',')

print(eegs.shape)

