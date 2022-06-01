
import numpy as np
import os

base_path = '/home/hyoseok/research/eeg/dataset/deap/deap_1s_norm/'

num_subject = 40
num_run= 16
num_trial = 1000

total_max = -10000000000
total_min = 10000000000
sum_max = 0
sum_min = 0
cnt = 0

for s in range(1, num_subject+1):
    if s in np.array([12, 21, 22, 23, 24, 33]):
        continue
    for r in range(1, num_run+1):
        for t in range(1, num_trial+1):
            data_name = 'S%02dT%02d_%04d.npy'%(s, r, t)
            file_name = base_path + data_name

            if not os.path.exists(file_name):
                break

            data = np.load(file_name)
            data_max = np.max(data)
            data_min = np.min(data)

            sum_max = sum_max + data_max
            sum_min = sum_min + data_min
            cnt = cnt + 1

            # print(data.shape)


            if data_max > total_max:
                total_max = data_max

            if data_min < total_min:
                total_min = data_min

    if cnt > 0:
        print('subject:%d mean max: %.2f mean min: %.2f'%(s, sum_max / cnt, sum_min / cnt))

print('max:%.2f min:%.2f'%(total_max, total_min))