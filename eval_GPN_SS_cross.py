#  20210713

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
import os
import sys
import configparam
import time
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
import pickle

from models import *
from adversarial_models import *
from lost_functions import *
from dataloaders.amigos_cnn_loader import amigos_cnn_loader
from dataloaders.deap_cnn_loader import deap_cnn_loader
from dataloaders.physionet_cnn_loader import physionet_cnn_loader
from dataloaders.ner2015_cnn_loader import ner2015_cnn_loader
from dataloaders.data_split import data_split

from sklearn.model_selection import KFold, train_test_split
k_folds = 5
torch.manual_seed(0)


def evaluation(param):
    param.PrintConfig()
    batch_size = param.batch_size

    perturbation_generating_model = param.model
    victim_model_list = ['eegnet', 'dconvnet', 'sconvnet', 'resnet', 'vgg', 'tidnet']

    # K-fold iteration
    k_folds = 5

    # Load Dataset
    if param.dataset == 'amigos':
        data_set = amigos_cnn_loader(param)
    elif param.dataset == 'deap':
        data_set = deap_cnn_loader(param)
    elif param.dataset == 'physionet':
        data_set = physionet_cnn_loader(param)
    elif param.dataset == 'ner2015':
        data_set = ner2015_cnn_loader(param)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # For fold results
    eval_results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):
        result = []
        for i in range(len(victim_model_list)):

            victim_model = victim_model_list[i]
            pretrained_name = param.pretrained_name.replace(param.model, victim_model)
            pretrained_weight_file = param.result_path.replace(param.model,
                                                               victim_model) + '/pretrained/' + f'fold{fold}_' + pretrained_name

            # Print fold num
            print('-----------------------')
            print(f'FOLD {fold}')
            print('-----------------------')

            # set victim model
            if victim_model == 'eegnet':
                print('Model: EEGNet')
                model = EEGNet(param.num_channel, param.num_length, param.num_class)
            elif victim_model == 'sconvnet':
                print('Shallow Conv Net')
                model = ShallowConvNet(param.num_channel, param.num_length, param.num_class)
            elif victim_model == 'dconvnet':
                print('Deep Conv Net')
                model = DeepConvNet(param.num_channel, param.num_length, param.num_class)
            elif victim_model == 'resnet':
                print('ResNet')
                model = ResNet8(param.num_class)
            elif victim_model == 'tidnet':
                print('TIDNet')
                model = TIDNet(in_chans = param.num_channel, n_classes = param.num_class, input_window_samples=param.num_length)
            elif victim_model == 'vgg':
                print('VGG')
                model = vgg_eeg(pretrained=False, num_classes=param.num_class)

            # If model not pretrained, quit
            if param.use_pretrained == 0:
                print('use pretrained has to be 1')
                exit()

            # Load victim model's pre-trained weight for each fold
            # pretrained_name: amigos_eegnet_e0200.pth
            print(pretrained_weight_file)
            model.load_state_dict(torch.load(pretrained_weight_file))

            model.eval()
            model.cuda()

            # Sample elements randomly from a given list of ids, no replacement.
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for testing data in this fold
            test_loader = torch.utils.data.DataLoader(data_set, batch_size=param.batch_size, sampler=test_subsampler, num_workers=12)

            # Save best perturbation
            if param.attack_type == 'non-targeted':
                save_file_name = '/home/airlab/Desktop/EEG/code/eeg_uap_airlab/result/' + param.dataset + '_'+ perturbation_generating_model + '/uap/0.0392/' + 'air_uap_net_nt_fold%d.pth' % fold

                # load UAP generator and discriminator
                generator = GenResNet(1, param.num_channel, param.num_length)
                generator.load_state_dict(torch.load(save_file_name))
                print('Load pretrained generator weight from: ', save_file_name)
                generator.eval()
                generator.cuda()

                # Constraint on magnitude of perturbation
                norm_type = param.norm_type
                norm_limit = param.epsilon

                num_positive = 0
                num_total = 0
                num_adv_positive = 0
                num_fool = 0

                for test_x, test_y in test_loader:

                    #  UAP Net evaluation
                    test_x = test_x.cuda()

                    adv_exam_cuda_test = generator(test_x)
                    norm_exam = adv_exam_cuda_test.view(adv_exam_cuda_test.shape[0], -1)
                    if norm_type == 'inf':
                        norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                    elif norm_type == 'L2':
                        norm_exam = torch.norm(norm_exam, p=2)
                    adv_exam_cuda = torch.mul(adv_exam_cuda_test / norm_exam.view(adv_exam_cuda_test.shape[0], 1, 1, 1),
                                              norm_limit)

                    # Set target class
                    if param.attack_type == 'targeted':
                        test_y = torch.add(torch.mul(test_y, 0), param.attack_target)

                    # Set label for each attack_type
                    if param.attack_type == 'non-targeted':
                        target_label = test_y.cuda()
                    elif param.attack_type == 'targeted':
                        target_label = torch.add(torch.mul(test_y, 0), param.attack_target).cuda()

                    # Get clean
                    with torch.no_grad():
                        output = model.forward(test_x)
                        output_sm = F.softmax(output, dim=1)
                        _, output_index = torch.max(output_sm, 1)
                        res = output_index.cpu().detach().numpy()

                    tp = (res == target_label.cpu().detach().numpy()).sum()

                    num_positive = num_positive + tp
                    num_total = num_total + res.shape[0]

                    # Add perturbation
                    test_x_adv = torch.add(test_x, adv_exam_cuda)

                    # Do clamping per channel
                    for cii in range(param.num_channel):
                        test_x_adv[:, :, cii, :] = test_x_adv[:, :, cii, :].clone().clamp(
                            min=test_x[:, :, cii, :].min(),
                            max=test_x[:, :, cii, :].max())

                    with torch.no_grad():
                        output = model.forward(test_x_adv)
                        output_sm = F.softmax(output, dim=1)
                        _, output_index = torch.max(output_sm, 1)
                        res_adv = output_index.cpu().detach().numpy()

                    tp = (res_adv == target_label.cpu().detach().numpy()).sum()
                    num_adv_positive = num_adv_positive + tp

                    num_fool += (res != res_adv).sum()

            else:

                num_positive = 0
                num_total = 0
                num_adv_positive = 0
                num_fool = 0

                for attack_target in range(param.num_class):
                    save_file_name = param.uap_path + 'air_uap_net_t%d_fold%d.pth' % (attack_target, fold)

                    # load UAP generator and discriminator
                    generator = GenResNet(1, param.num_channel, param.num_length)
                    generator.load_state_dict(torch.load(save_file_name))
                    print('Load pretrained generator weight from: ', save_file_name)
                    generator.eval()
                    generator.cuda()

                    # Constraint on magnitude of perturbation
                    norm_type = param.norm_type
                    norm_limit = param.epsilon


                    for test_x, test_y in test_loader:

                        #  UAP Net evaluation
                        test_x = test_x.cuda()

                        adv_exam_cuda_test = generator(test_x)
                        norm_exam = adv_exam_cuda_test.view(adv_exam_cuda_test.shape[0], -1)
                        if norm_type == 'inf':
                            norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                        elif norm_type == 'L2':
                            norm_exam = torch.norm(norm_exam, p=2)
                        adv_exam_cuda = torch.mul(adv_exam_cuda_test / norm_exam.view(adv_exam_cuda_test.shape[0], 1, 1, 1),
                                                  norm_limit)
                        # Set target label
                        target_label = torch.add(torch.mul(test_y, 0), attack_target).cuda()

                        # Get clean
                        with torch.no_grad():
                            output = model.forward(test_x)
                            output_sm = F.softmax(output, dim=1)
                            _, output_index = torch.max(output_sm, 1)
                            res = output_index.cpu().detach().numpy()

                        tp = (res == target_label.cpu().detach().numpy()).sum()

                        num_positive = num_positive + tp
                        num_total = num_total + res.shape[0]

                        # Add perturbation
                        test_x_adv = torch.add(test_x, adv_exam_cuda)

                        # Do clamping per channel
                        for cii in range(param.num_channel):
                            test_x_adv[:, :, cii, :] = test_x_adv[:, :, cii, :].clone().clamp(min=test_x[:, :, cii, :].min(),
                                                                                              max=test_x[:, :, cii, :].max())

                        with torch.no_grad():
                            output = model.forward(test_x_adv)
                            output_sm = F.softmax(output, dim=1)
                            _, output_index = torch.max(output_sm, 1)
                            res_adv = output_index.cpu().detach().numpy()

                        tp = (res_adv == target_label.cpu().detach().numpy()).sum()
                        num_adv_positive = num_adv_positive + tp

                        num_fool += (res != res_adv).sum()

            clean_test_accuracy = num_positive / num_total
            adv_test_accuracy = num_adv_positive / num_total
            fooling_ratio = num_fool / num_total

            print('Perturbation generating model:', perturbation_generating_model)
            print('Victim model:', victim_model)
            print('test accuracy: %.4f -> %.4f ( %d / %d)'%(clean_test_accuracy, adv_test_accuracy, num_positive, num_total))
            print('fooling rate: %.4f'%(fooling_ratio))
            print('---')

            result.append([clean_test_accuracy, adv_test_accuracy, fooling_ratio])

        eval_results.append(result)

    print('--------------------------------')
    print(f'Finished K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('Perturbation Generating Model:', perturbation_generating_model)

    print(np.array(eval_results).shape)

    print('-----------------')
    print('Final results')

    for i in range(len(victim_model_list)):
        sum_clean = 0.0
        sum_adv = 0.0
        sum_fool = 0.0
        for ii in range(k_folds):
            sum_clean += eval_results[ii][i][0]
            sum_adv += eval_results[ii][i][1]
            sum_fool += eval_results[ii][i][2]
        print('Average on %s: %.4f -> %.4f'%(victim_model_list[i], sum_clean/k_folds, sum_adv/k_folds))
        print('Fooling ratio: %.4f'%(sum_fool / k_folds))


if __name__ == '__main__':

    no_gpu = 2

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/target0/eval_amigos_eegnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d'%no_gpu)

    evaluation(conf)

