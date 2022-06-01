import numpy as np
from scipy import io
import os


preprocessed_mat_path = '/home/hyoseok/research/eeg/dataset/amigos/data_preprocessed_matalb/'
preprocessed_python_path = '/home/hyoseok/research/eeg/dataset/amigos/data_preprocessed_python/'

if not os.path.exists(preprocessed_python_path):
    os.makedirs(preprocessed_python_path)

num_subject = 40
# for short vidoes only
num_trial = 16

# num_subject = 1
# num_trial = 1

for i in range(1, num_subject+1):
    mat_file_path = preprocessed_mat_path + 'Data_Preprocessed_P%02d'%i + '/'
    python_file_path = preprocessed_python_path

    mat_file_eeg = mat_file_path + 'Data_Preprocessed_P%02d.mat'%i
    print(mat_file_eeg)
    mat_file = io.loadmat(mat_file_eeg)
    print(mat_file.keys())
    eeg_mat = mat_file['joined_data']
    label_self_mat = mat_file['labels_selfassessment']
    label_ext_mat = mat_file['labels_ext_annotation']

    # original eeg dataset(17): 1~14: eeg / 15, 16: EOG / 17:GSR
    # Only EEG signals are extracted and transpose
    for j in range(1, num_trial+1):
        eeg_trial_mat = np.transpose(eeg_mat[0, j-1][:, :14])
        label_self = label_self_mat[0,j-1]
        label_ext = label_ext_mat[0, j - 1]

        python_file_eeg = python_file_path + 'S%02dT%02d.npy'%(i, j)
        python_file_label_self = python_file_path + 'S%02dT%02d_label_self.txt'%(i, j)
        python_file_label_ext = python_file_path + 'S%02dT%02d_label_ext.txt' % (i, j)

        print(python_file_eeg)
        print(python_file_label_self)
        print(python_file_label_ext)
        np.save(python_file_eeg, eeg_trial_mat)
        np.savetxt(python_file_label_self, label_self, delimiter=' ', fmt='%.4f')
        np.savetxt(python_file_label_ext, label_ext, delimiter=' ', fmt='%.4f')






