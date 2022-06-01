import numpy as np
import os

num_sub = 109
num_trial = 14
sampling_rate = 160
base_source_path = '/home/airlab/Desktop/EEG/dataset/physionet/physionet_4s_norm/'
base_target_path = '/home/airlab/Desktop/EEG/dataset/physionet/physionet_1s_norm/'

if not os.path.exists(base_target_path):
    os.makedirs(base_target_path)

except_list = [2, 3, 5, 7, 9, 11, 13]
except_sub_list = [88, 92, 100, 104]

for s in range(num_sub):
    if s+1 in except_sub_list:
        continue
    for v in range(num_trial):
        #  we don't know exact length of each trial. So, if the npy file is not exist, skip to next trial.
        if v+1 in except_list:
            continue
        for t in range(14):
            eeg_4s_name = base_source_path+'S%03dR%02dT%03d.npy'%(s+1,v+1,t+1)
            label_4s_name = base_source_path+'S%03dR%02dT%03d_label.txt'%(s+1,v+1,t+1)

            # Split 4s norm into 1s norm
            for sub in range(1,5):
                eeg_1s_name = base_target_path+'S%03dR%02dT%03d.npy'%(s+1, v+1, 4*t+sub)
                label_1s_name = base_target_path+'S%03dR%02dT%03d_label.txt'%(s+1, v+1, 4*t+sub)

                f = open(label_4s_name, 'r')
                val = float(f.read().replace('\n', ''))

                eeg_sub_data = np.load(eeg_4s_name)
                eeg_sub_data = eeg_sub_data[sampling_rate*sub:sampling_rate*(sub+1)]
                print('saved' + eeg_1s_name)
                np.save(eeg_1s_name, eeg_sub_data)
                np.savetxt(label_1s_name, [val])