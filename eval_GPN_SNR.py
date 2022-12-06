# 20220802
# Gaussian noise

import sys
import configparam
import numpy as np

from models import *
from adversarial_models import *
from dataloaders.amigos_cnn_loader_subj import amigos_cnn_loader
from dataloaders.deap_cnn_loader_subj import deap_cnn_loader
from dataloaders.physionet_cnn_loader_subj import physionet_cnn_loader
from dataloaders.ner2015_cnn_loader_subj import ner2015_cnn_loader

from sklearn.model_selection import KFold
from adversarial_models.GenResNetHyper import *

np.random.seed(0)
torch.manual_seed(0)
k_folds = 5

def signaltonoise_dB(a, n, axis=0, ddof=0):
    a = np.asanyarray(a)

    ps = np.abs(np.fft.fft2(a) ** 2)
    pn = np.abs(np.fft.fft2(n) ** 2)

    return 10 * np.log10(np.average(ps/pn))
    # m = a.mean(axis)
    # sd = a.std(axis=axis, ddof=ddof) + 0.0001
    # return np.average(20*np.log10(abs(np.where(sd == 0, 0, m/sd))))

def evaluation(param):
    param.PrintConfig()
    batch_size = param.batch_size

    # set model
    if param.model == 'eegnet':
        print('Model: EEGNet')
        model = EEGNet(param.num_channel, param.num_length, param.num_class)
    elif param.model == 'sconvnet':
        print('Shallow Conv Net')
        model = ShallowConvNet(param.num_channel, param.num_length, param.num_class)
    elif param.model == 'dconvnet':
        print('Deep Conv Net')
        model = DeepConvNet(param.num_channel, param.num_length, param.num_class)
    elif param.model == 'resnet':
        print('ResNet')
        model = ResNet8(param.num_class)
        # model = EEGResNet(in_chans=param.num_channel, n_classes=param.num_class, input_window_samples=param.num_length)
    elif param.model == 'tidnet':
        print('TIDNet')
        model = TIDNet(in_chans = param.num_channel, n_classes = param.num_class, input_window_samples=param.num_length)
    elif param.model == 'vgg':
        print('VGG')
        model = vgg_eeg(pretrained=False, num_classes=param.num_class)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # For fold results
    results = []

    subject_list = [i for i in range(param.num_subject)]

    for fold, (train_ids, test_ids) in enumerate(kfold.split(subject_list)):
        # Print
        print('-----------------------')
        print(f'FOLD {fold}')
        print('-----------------------')


        # Load dataset!
        if param.dataset == 'amigos':
            train_dataset = amigos_cnn_loader(param, subject_list=train_ids)
            test_dataset = amigos_cnn_loader(param, subject_list=test_ids)
        elif param.dataset == 'deap':
            train_dataset = deap_cnn_loader(param, subject_list=train_ids)
            test_dataset = deap_cnn_loader(param, subject_list=test_ids)
        elif param.dataset == 'physionet':
            train_dataset = physionet_cnn_loader(param, subject_list=train_ids)
            test_dataset = physionet_cnn_loader(param, subject_list=test_ids)
        elif param.dataset == 'ner2015':
            train_dataset = ner2015_cnn_loader(param, subject_list=train_ids)
            test_dataset = ner2015_cnn_loader(param, subject_list=test_ids)

        # Define data loaders for training and testing data in this fold
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False,num_workers=12)

        # If not pretrained, quit
        if param.use_pretrained == 0:
            print('use pretrained has to be 1')
            exit()

        # Load model
        # pretrained_weight_file = param.result_path + '/pretrained/' + f'fold{fold}_' + param.pretrained_name # Within-subject
        # pretrained_weight_file = param.weight_path + f'fold{fold}_' + param.weight_prefix + '_e{:04d}_subj.pth'.format(50)
        pretrained_weight_file = param.result_path + '/pretrained/' + f'fold{fold}_' + param.pretrained_name
        # pretrained_weight_file = param.result_path + '/pretrained/' + f'fold{fold}_' + '_subj' + param.pretrained_name # Leave Subject
        print('Load pretrained Model:', pretrained_weight_file)
        model.load_state_dict(torch.load(pretrained_weight_file))

        model.eval()
        model.cuda()

        generator = GenResNet(1, param.num_channel, param.num_length)
        save_file_name = param.uap_path + 'air_uap_net_nt_fold%d.pth' % fold
        generator.load_state_dict(torch.load(save_file_name))  # If there's pretrained weight
        generator.eval()
        generator.cuda()

        # Reset for test
        clean_num_positive = 0
        clean_num_total = 0
        num_positive = 0
        num_total = 0
        num_fool = 0
        SNR_db = 0

        for test_x, test_y in test_loader:

            # Generate random noise
            # adv_exam_cuda_perturbation = 0.0392 * np.random.uniform(-1, 1, (1, param.num_channel, param.num_length)).astype(np.float32)

            # DF
            # uap_file_name = param.uap_path + 'df_uap_fold%d_subj.npy'%fold  # DF
            # adv_exam_cuda_perturbation = np.load(uap_file_name)

            # TLM
            # uap_file_name = param.uap_path + '_uap_tlm_non_targeted_fold%d_subj.npy'%fold # TLM
            # adv_exam_cuda_perturbation = np.load(uap_file_name)

            # GPN-SA
            # uap_file_name = param.uap_path + 'uap_air_exam_nt_fold%d_subj.npy' % fold
            # adv_exam_cuda_perturbation = np.load(uap_file_name)

            # GPN-SS
            generator.zero_grad()
            adv_exam_cuda_perturbation = generator(test_x.cuda())

            # adv_exam_cuda_perturbation = torch.from_numpy(adv_exam_cuda_perturbation).cuda()
            # adv_exam_cuda_perturbation = adv_exam_cuda_perturbation.cuda()

            # load UAP generator and discriminator
            # generator = GenResNet(1, param.num_channel, param.num_length)
            # generator.load_state_dict(torch.load(save_file_name))
            # print('Load pretrained generator weight from: ', save_file_name)
            # generator.eval()
            # generator.cuda()

            test_x_adv = torch.add(test_x.cuda(), adv_exam_cuda_perturbation)

            # Do clamping per channel
            for cii in range(param.num_channel):
                test_x_adv[:, :, cii, :] = test_x_adv[:, :, cii, :].cpu().clone().clamp(min=test_x[:, :, cii, :].min(),
                                                                                  max=test_x[:, :, cii, :].max())

            if param.attack_type == 'targeted':
                test_y = torch.add(torch.mul(test_y, 0), param.attack_target)

            with torch.no_grad():
                # Clean Accuracy
                output = model.forward(test_x.cuda())
                output_sm = F.softmax(output, dim=1)
                _, pred_label = torch.max(output_sm, 1)
                clean_res_test = pred_label.cpu().detach().numpy()

                # Adversarial Accuracy
                output = model.forward(test_x_adv)
                output_sm = F.softmax(output, dim=1)
                _, output_index = torch.max(output_sm, 1)
                res_test = output_index.cpu().detach().numpy()

            clean_tp_test = (clean_res_test == test_y.detach().numpy()).sum()
            tp_test = (res_test == test_y.detach().numpy()).sum()

            clean_num_positive = clean_num_positive + clean_tp_test
            num_positive = num_positive + tp_test
            num_fool += (res_test != pred_label.cpu().detach().numpy()).sum()
            num_total = num_total + res_test.shape[0]
            SNR_db += signaltonoise_dB(test_x_adv.cpu().detach().numpy(), adv_exam_cuda_perturbation.cpu().detach().numpy())

        clean_test_accuracy = clean_num_positive / num_total
        test_accuracy = num_positive / num_total
        test_fooling_ratio = num_fool / num_total
        SNR_db /= len(test_loader)

        results.append([clean_test_accuracy, test_accuracy, test_fooling_ratio, SNR_db])

        print('Adversarial test result on fold {}: {:.4f} -> {:.4f}, test fooling ratio {:.4f}, SNR: {:.4f}'.format(fold,
                                                                                                       clean_test_accuracy,
                                                                                                       test_accuracy,
                                                                                                       test_fooling_ratio,
                                                                                                                    SNR_db))

    # Print fold results
    print(f'Finished K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_clean = 0.0
    sum_adv = 0.0
    sum_fool = 0.0
    snr = 0.0
    for i in range(len(results)):
        print(
            'Fold : {}, test_acc : {:.4f} -> {:.4f}, test fooling ratio {:.4f}, SNR: {:.4f}'.format(i, results[i][0], results[i][1],
                                                                                       results[i][2], results[i][3]))
        sum_clean += results[i][0]
        sum_adv += results[i][1]
        sum_fool += results[i][2]
        snr += results[i][3]
    print('Average: {:.4f} -> {:.4f}, fooling ratio {:.4f}, SNR: {:.4f}'.format(sum_clean / len(results), sum_adv / len(results),
                                                                   sum_fool / len(results), snr / len(results)))

if __name__ == '__main__':

    no_gpu = 7

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_physionet_tidnet.cfg'
        # conf_file_name = './config/train_amigos_sconvnet.cfg'
        # conf_file_name = './config/train_amigos_dconvnet.cfg'
        # conf_file_name = './config/eval_amigos_resnet.cfg'
        # conf_file_name = './config/train_amigos_tidnet.cfg'
        # conf_file_name = './config/train_amigos_newnet.cfg'
        # conf_file_name = './config/eval_deap_eegnet.cfg'
        # conf_file_name = './config/train_deap_resnet.cfg'
        # conf_file_name = './config/train_physionet_eegnet.cfg'
        # conf_file_name = './config/train_ner2015_eegnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d'%no_gpu)

    evaluation(conf)

