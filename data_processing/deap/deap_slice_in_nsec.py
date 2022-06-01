import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

channels = 32
sampling_rate = 128
segment_size = 1      # sec
segment_length = sampling_rate * segment_size

normalization = True
#  normal method: 1 - Z-score / 2 - min-max / 3 - Z-score (0~1)
normal_method = 3

num_subject = 32
num_trial = 40

base_source_path = '/home/hyoseok/research/eeg/dataset/deap/data_preprocessed_python/'
base_target_path = '/home/hyoseok/research/eeg/dataset/deap/deap_1s_norm/'

if not os.path.exists(base_target_path):
    os.mkdir(base_target_path)

for i in range(1, num_subject+1):
    print(base_source_path + 's%02d.dat'%i)
    with open(base_source_path + 's%02d.dat'%i, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')

    eegs_subj = dataset['data'][:, :32, :]
    labels_subj = dataset['labels']

    for j in range(1, num_trial+1):

        eegs = np.squeeze(eegs_subj[j-1, :, :])
        labels = labels_subj[j-1, :]

        if normalization == True:
            if normal_method  == 1:
                mean = np.transpose(np.mean(eegs, axis=1).reshape(-1, eegs.shape[0]))
                stdev = np.transpose(np.std(eegs, axis=1).reshape(-1, eegs.shape[0]))
                eegs = (eegs - mean) / (stdev * 2)
            elif normal_method == 3:
                mean = np.transpose(np.mean(eegs, axis=1).reshape(-1, eegs.shape[0]))
                stdev = np.transpose(np.std(eegs, axis=1).reshape(-1, eegs.shape[0]))
                eegs = (eegs - mean) / (stdev * 8) + 0.5
                eegs = np.clip(eegs, 0, 1)

        # for kk in range(eegs.shape[0]):
        #     plt.plot(eegs[kk, :])
        # plt.show()

        valence = [labels[0]]
        arousal = [labels[1]]
        multi = -1

        if valence[0] >= 5.0:
            if arousal[0] >= 5.0:
                multi = 0
            else:
                multi = 3
        else:
            if arousal[0] >= 5.0:
                multi = 1
            else:
                multi = 2

        num_segment = int(eegs.shape[1] / segment_length)

        for k in range(1, num_segment+1):
            eeg = eegs[:, segment_length*(k-1): segment_length*k]
            eeg_segment_file = base_target_path + 'S%02dT%02d_%04d.npy'%(i, j, k)
            label_segment_valence_file = base_target_path + 'S%02dT%02d_%04d_valence.txt' % (i, j, k)
            label_segment_arousal_file = base_target_path + 'S%02dT%02d_%04d_arousal.txt' % (i, j, k)
            label_segment_multi_file = base_target_path + 'S%02dT%02d_%04d_multi.txt' % (i, j, k)

            np.save(eeg_segment_file, eeg)
            np.savetxt(label_segment_valence_file, valence, delimiter=' ', fmt='%.4f')
            np.savetxt(label_segment_arousal_file, arousal, delimiter=' ', fmt='%.4f')
            with open(label_segment_multi_file, 'w') as f:
                f.write('%d' % multi)







