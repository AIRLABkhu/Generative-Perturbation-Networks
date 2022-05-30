import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
import sys
import configparam
from torch.utils.data.sampler import SubsetRandomSampler

from adversarial_models import *
from lost_functions import *
from dataloaders.amigos_cnn_loader import amigos_cnn_loader

from sklearn.model_selection import KFold, train_test_split

k_folds = 5
torch.manual_seed(0)


def evaluation(param):
    param.PrintConfig()

    fig, ax = plt.subplots(ncols=1)
    biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    n_channels = len(ch_names)
    ch_types = ['eeg'] * 14
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=128)

    # K-fold iteration
    k_folds = 5

    # Load Dataset
    data_set = amigos_cnn_loader(param)

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):
        if fold > 0:
            break

        # Sample elements randomly from a given list of ids, no replacement.
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for testing data in this fold
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=param.batch_size, sampler=test_subsampler, num_workers=12)

        #save_file_name = '/home/airlab/Desktop/EEG/code/eeg_uap_airlab/result/amigos_eegnet/uap/0.0392/air_uap_net_nt_fold0.pth'
        save_file_name = '/home/airlab/Desktop/EEG/code/eeg_uap_airlab/result/amigos_eegnet/uap/0.0392/air_uap_net_t0_fold0.pth'


        # load UAP generator and discriminator
        generator = GenResNet(1, param.num_channel, param.num_length)
        generator.load_state_dict(torch.load(save_file_name))
        print('Load pretrained generator weight from: ', save_file_name)
        generator.eval()
        generator.cuda()

        ii = 0
        print(len(test_loader))
        for test_x, test_y in test_loader:
            if ii == (len(test_loader) // 2):
                data = test_x
                perturbation = generator(test_x.cuda())
                norm_exam = perturbation.view(perturbation.shape[0], -1)
                norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                perturbation = torch.mul(perturbation / norm_exam.view(perturbation.shape[0], 1, 1, 1), 0.0392)
                break
            ii += 1


        data = data[20].numpy()
        data = data.squeeze()

        print(np.min(data), np.max(data))
        perturbation = perturbation[20].cpu().detach().numpy()
        perturbation = perturbation.squeeze()

        adv_data = data + perturbation
        adv_data = np.clip(adv_data, 0, 1)
        print(np.min(adv_data), np.max(adv_data))

        print(data.shape)
        print(perturbation.shape)

        evoked = mne.EvokedArray(data, info)
        evoked.set_montage(biosemi_montage)
        montage_head = evoked.get_montage()
        ch_pos = montage_head.get_positions()['ch_pos']
        pos = np.stack([ch_pos[ch] for ch in ch_names])

        im1, cm1 = mne.viz.plot_topomap(evoked.data[:, 15], evoked.info, show=False, cmap='jet', vmin=0, vmax=np.max(data))

        evoked_adv = mne.EvokedArray(adv_data, info)
        evoked_adv.set_montage(biosemi_montage)
        montage_head = evoked_adv.get_montage()
        ch_pos = montage_head.get_positions()['ch_pos']
        pos = np.stack([ch_pos[ch] for ch in ch_names])

        #im2, cm2 = mne.viz.plot_topomap(evoked_adv.data[:, 15], evoked.info, show=False, cmap='jet', vmin=0, vmax=np.max(adv_data))

        #fig.colorbar(im1)
        cb = fig.colorbar(im1)
        cb.ax.tick_params(labelsize=15)


        #ax.set_title('Benign examples')

        #ax.set_title('Adversarial examples')
        plt.show()

if __name__ == '__main__':

    no_gpu = 0

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = '/home/airlab/Desktop/EEG/code/eeg_uap_airlab/config/non-target/eval_amigos_eegnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d'%no_gpu)

    evaluation(conf)

