# 20220119

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

torch.manual_seed(0)
k_folds = 5

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1: # Conv가 존재시
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1: # BatchNorm이 존재시
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def train(param):
    param.PrintConfig()
    learning_rate = param.learning_rate
    num_epoch = param.num_epoch
    patience = 15

    # loading model and dataset
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
        model = TIDNet(in_chans=param.num_channel, n_classes=param.num_class, input_window_samples=param.num_length)
    elif param.model == 'vgg':
        print('VGG')
        model = vgg_eeg(pretrained=False, num_classes=param.num_class)

    # Load dataset!
    if param.dataset == 'amigos':
        data_set = amigos_cnn_loader(param)
    elif param.dataset == 'deap':
        data_set = deap_cnn_loader(param)
    elif param.dataset == 'physionet':
        data_set = physionet_cnn_loader(param)
    elif param.dataset == 'ner2015':
        data_set = ner2015_cnn_loader(param)

    if param.use_predefined_idx == 0:
        print('pretrained index has to be 1')
        exit()

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # For fold results
    results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):

        # Print
        print('-----------------------')
        print(f'FOLD {fold}')
        print('-----------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        # train_ids, val_ids = train_test_split(train_ids, test_size=0.25, shuffle=True, random_state=0)

        np.random.seed(0)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(data_set, batch_size=param.batch_size, sampler=train_subsampler, num_workers=12)
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=param.batch_size, sampler=test_subsampler, num_workers=12)

        # If not pretrained, quit
        if param.use_pretrained == 0:
            print('use pretrained has to be 1')
            exit()

        # Load model
        pretrained_weight_file = param.result_path + '/pretrained/' + f'fold{fold}_' + param.pretrained_name
        print('Load pretrained Model:' + pretrained_weight_file)
        model.load_state_dict(torch.load(pretrained_weight_file))

        model.eval()
        model.cuda()

        # load UAP generator and discriminator
        generator = GenResNet(1, param.num_channel, param.num_length)
        generator.apply(weights_init)
        generator.train()
        generator.cuda()

        # Universal Example is from noise that has same size with the input signals
        # fix distributions of noise
        np.random.seed(0)
        init_noise = np.random.uniform(0, 1, (param.num_channel, param.num_length))
        init_noise = np.reshape(init_noise, (1, param.num_channel, param.num_length))
        init_noise = init_noise[np.newaxis, :, :, :]
        # init_noise = np.tile(init_noise, (param.batch_size, 1, 1, 1))
        init_noise_cuda = torch.from_numpy(init_noise).type(torch.FloatTensor).cuda()
        # init_noise_batch = np.tile(init_noise, (param.batch_size, 1, 1, 1))
        # init_noise_cuda = torch.FloatTensor(init_noise_batch).cuda()

        # Define Loss function
        loss_func = nn.CrossEntropyLoss()
        #loss_func = FocalLoss()

        # Define Adam optimizer and scheduler
        optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # Constraint on magnitude of perturbation
        norm_type = param.norm_type
        norm_limit = param.epsilon

        # Set best_acc
        if param.attack_type == 'non-targeted':
            best_val_accuracy = 100
        elif param.attack_type == 'targeted':
            best_val_accuracy = 0

        loss_total = 0.0
        cnt = 0

        for i in range(num_epoch):
            loss_epoch = 0.0
            cnt_epoch = 0

            num_positive = 0
            num_total = 0
            num_fool = 0

            t0 = time.time()
            train_fooling_ratio = 0
            test_fooling_ratio = 0

            for train_x, train_y in train_loader:

                train_x = train_x.cuda()
                train_y = train_y.cuda()

                if param.attack_type == 'non-targeted':
                    # target_label = train_y.cuda()
                    cur_noise_cuda = init_noise_cuda

                    # Get original prediction from victim model
                    with torch.no_grad():
                        output = model.forward(train_x.cuda())
                        output_sm = F.softmax(output, dim=1)
                        _, target_label = torch.max(output_sm, 1)

                elif param.attack_type == 'targeted':
                    target_label = torch.add(torch.mul(train_y, 0), param.attack_target)
                    target_label = target_label.cuda()
                    cur_noise_cuda = init_noise_cuda

                generator.zero_grad()

                #  Generate perturbation
                adv_exam_cuda = generator(cur_noise_cuda) # range : -1 ~ 1
                # adv_exam_cuda_perturbation = adv_exam_cuda * norm_limit  # range : -norm_limit ~ norm_limit
                # Scale
                norm_exam = adv_exam_cuda.view(adv_exam_cuda.shape[0], -1)
                if norm_type == 'inf':
                    norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                elif norm_type == 'L2':
                    norm_exam = torch.norm(norm_exam, p=2, dim=1)
                adv_exam_cuda = torch.mul(adv_exam_cuda / norm_exam.view(adv_exam_cuda.shape[0], 1, 1, 1), norm_limit)
                
                train_x_adv = torch.add(train_x.cuda(), adv_exam_cuda_perturbation)

                # Do clamping per channel
                for cii in range(param.num_channel):
                    train_x_adv[:, :, cii, :] = train_x_adv[:, :, cii, :].clone().clamp(min=train_x[:, :, cii, :].min(), max=train_x[:, :, cii, :].max())

                output = model.forward(train_x_adv)
                if param.attack_type == 'non-targeted':
                    loss = torch.log(loss_func(1-F.softmax(output, dim=1), target_label))
                    #loss = -torch.log(loss_func(output, target_label))
                    #loss = loss_func(-output, target_label)

                elif param.attack_type == 'targeted':
                    loss = loss_func(output, target_label)

                loss.backward()
                optimizer.step()

                # Train acc
                output_sm = F.softmax(output, dim=1)
                _, output_index = torch.max(output_sm, 1)
                res = output_index.cpu().detach().numpy()
                if param.attack_type == 'non-targeted':
                    tp = (res == train_y.cpu().detach().numpy()).sum()
                elif param.attack_type == 'targeted':
                    tp = (res == target_label.cpu().detach().numpy()).sum()

                num_positive = num_positive + tp
                num_total = num_total + res.shape[0]
                # Fooling rate
                num_fool += (res != target_label.cpu().detach().numpy()).sum()

            scheduler.step()
            train_accuracy = num_positive / num_total
            train_fooling_ratio = num_fool / num_total

            num_positive = 0
            num_total = 0
            cnt_epoch += 1
            uap_exam = adv_exam_cuda_perturbation.cpu().detach().numpy()[0, :, :, :]

            t1 = time.time()
            print(
                'epoch:{} loss:{:.4f} train accuracy:{:.4f} train fooling ratio:{:.4f} time:{:.4f} lr:{}'.format(
                    i + 1, loss,  train_accuracy, train_fooling_ratio, (t1 - t0),
                    scheduler.get_last_lr()))

            # Save best perturbation
            # if param.attack_type == 'non-targeted':
            #     uap_file_name = param.uap_path + 'uap_air_exam_nt_fold.npy'%fold
            #     np.save(uap_file_name, uap_exam)
            #     print('Saved best perturbation at ' + uap_file_name)
            # else:
            #     uap_file_name = param.uap_path + 'uap_air_exam_t%d_fold%d.npy' % (param.attack_target, fold)
            #     np.save(uap_file_name, uap_exam)
            #     print('Saved best perturbation at ' + uap_file_name)


        # Reset for test
        clean_num_positive = 0
        clean_num_total = 0
        num_positive = 0
        num_total = 0
        num_fool = 0

        for test_x, test_y in test_loader:

            #adv_exam_cuda_perturbation = np.load(uap_file_name)
            #adv_exam_cuda_perturbation = torch.from_numpy(adv_exam_cuda_perturbation).cuda()
            #adv_exam_cuda_perturbation = adv_exam_cuda_perturbation.cuda()

            #  Generate perturbation
            adv_exam_cuda = generator(cur_noise_cuda)  # range : -1 ~ 1
            adv_exam_cuda_perturbation = adv_exam_cuda * norm_limit  # range : -norm_limit ~ norm_limit
            test_x_adv = torch.add(test_x.cuda(), adv_exam_cuda_perturbation)

            # Do clamping per channel
            for cii in range(param.num_channel):
                test_x_adv[:, :, cii, :] = test_x_adv[:, :, cii, :].clone().clamp(min=test_x[:, :, cii, :].min(),
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

        clean_test_accuracy = clean_num_positive / num_total
        test_accuracy = num_positive / num_total
        test_fooling_ratio = num_fool / num_total

        results.append([clean_test_accuracy, test_accuracy, test_fooling_ratio])

        print('Adversarial test result on fold {}: {:.4f} -> {:.4f}, test fooling ratio {:.4f}'.format(fold, clean_test_accuracy, test_accuracy, test_fooling_ratio))

    # Print fold results
    print(f'Finished K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_clean = 0.0
    sum_adv = 0.0
    sum_fool = 0.0
    for i in range(len(results)):
        print('Fold : {}, test_acc : {:.4f} -> {:.4f}, test fooling ratio {:.4f}'.format(i, results[i][0], results[i][1], results[i][2]))
        sum_clean += results[i][0]
        sum_adv += results[i][1]
        sum_fool += results[i][2]
    print('Average: {:.4f} -> {:.4f}, fooling ratio {:.4f}'.format(sum_clean / len(results), sum_adv / len(results), sum_fool / len(results)))

    # Save result
    result_list = np.array(results)
    result_list = np.append(result_list, np.array([[sum_clean / len(results), sum_adv / len(results), sum_fool / len(results)]]), axis=0)

    # if param.attack_type == 'targeted':
    #     np.savetxt(param.uap_path + '_air_exam_result_target%d_fold.txt' % param.attack_target, result_list, fmt='%1.4f')
    #     print('saved at' + param.uap_path + '_air_exam_result_target%d_fold.txt'% param.attack_target)
    # elif param.attack_type == 'non-targeted':
    #     np.savetxt(param.uap_path + '_air_exam_result_non_target_fold.txt', result_list, fmt='%1.4f')
    #     print('saved at' + param.uap_path + '_air_exam_result_non_target_fold.txt')

if __name__ == '__main__':

    no_gpu = 1
    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_physionet_sconvnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d' % no_gpu)

    train(conf)
