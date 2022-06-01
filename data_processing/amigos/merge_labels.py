# make segment of subject - trial eeg dataset
import numpy as np
import matplotlib.pyplot as plt
import os

channels = 14
sampling_rate = 128
segment_size = 2      # sec
segment_length = sampling_rate * segment_size

normalization = True
#  normal method: 1 - Z-score / 2 - min-max
normal_method = 1

num_subject = 40
num_trial = 16

base_path = '/home/hyoseok/research/eeg/dataset/amigos/amigos_1s_norm/'
max_segment = 1000

# multi: 0 - HVHA / 1 - LVHA / 2 - LVLA / 3 - HVLA
multi = -1
stat = np.zeros((4), dtype=np.int32)

for i in range(1, num_subject+1):
    for j in range(1, num_trial+1):
        for k in range(1, max_segment+1):

            valence_name = base_path + 'S%02dT%02d_%04d_valence.txt'%(i, j, k)
            arousal_name = base_path + 'S%02dT%02d_%04d_arousal.txt'%(i, j, k)
            multi_name = base_path + 'S%02dT%02d_%04d_multi.txt' % (i, j, k)

            if not os.path.exists(valence_name):
                break


            valence = float(np.loadtxt(valence_name))
            arousal = float(np.loadtxt(arousal_name))

            if valence >= 5.0:
                if arousal >= 5.0:
                    multi = 0
                else:
                    multi = 3
            else:
                if arousal >= 5.0:
                    multi = 1
                else:
                    multi = 2

            stat[multi] = stat[multi]+1

            with open(multi_name, 'w') as f:
                f.write('%d' % multi)
            # print(multi_name)
            # print('%f %f %d'%(valence, arousal, multi))

    print(stat)







