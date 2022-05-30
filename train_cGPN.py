# 20220119
import random

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
from adversarial_models.GenResNetHyper import *
from lost_functions import *
from dataloaders.amigos_cnn_loader import amigos_cnn_loader
from dataloaders.deap_cnn_loader import deap_cnn_loader
from dataloaders.physionet_cnn_loader import physionet_cnn_loader
from dataloaders.ner2015_cnn_loader import ner2015_cnn_loader
from dataloaders.data_split import data_split

from torch.utils.data import ConcatDataset

from sklearn.model_selection import KFold
k_folds = 5
initial_weight = True
random.seed(0)

torch.manual_seed(0)
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
    batch_size = param.batch_size
    num_epoch = param.num_epoch

    res_list_test = np.array([]).reshape((0, 4))
    result = []
    start = iter

    # Set dataset
    if param.dataset == 'amigos':
        data_set = amigos_cnn_loader(param)
        data_idx = 0
    elif param.dataset == 'deap':
        data_set = deap_cnn_loader(param)
        data_idx = 1
    elif param.dataset == 'physionet':
        data_set = physionet_cnn_loader(param)
        data_idx = 2
    elif param.dataset == 'ner2015':
        data_set = ner2015_cnn_loader(param)
        data_idx = 3

    # Set Model
    model1 = EEGNet(param.num_channel, param.num_length, param.num_class)
    model2 = DeepConvNet(param.num_channel, param.num_length, param.num_class)
    model3 = ShallowConvNet(param.num_channel, param.num_length, param.num_class)
    model4 = ResNet8(param.num_class)
    model5 = TIDNet(in_chans=param.num_channel, n_classes=param.num_class, input_window_samples=param.num_length)
    model6 = vgg_eeg(pretrained=False, num_classes=param.num_class)


    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):

        # Print
        print('-----------------------')
        print(f'FOLD {fold}')
        print('-----------------------')

        np.random.seed(0)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(data_set, batch_size=param.batch_size, sampler=train_subsampler, num_workers=12)
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=param.batch_size, sampler=test_subsampler, num_workers=12)

        # Load pre-trained wight
        pretrained_weight_file1 = './result/' + param.dataset + '_eegnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_eegnet_e0050.pth'
        pretrained_weight_file2 = './result/' + param.dataset + '_dconvnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_dconvnet_e0050.pth'
        pretrained_weight_file3 = './result/' + param.dataset + '_sconvnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_sconvnet_e0050.pth'
        pretrained_weight_file4 = './result/' + param.dataset + '_resnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_resnet_e0050.pth'
        pretrained_weight_file5 = './result/' + param.dataset + '_tidnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_tidnet_e0050.pth'
        pretrained_weight_file6 = './result/' + param.dataset + '_vgg' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_vgg_e0050.pth'

        model1.load_state_dict(torch.load(pretrained_weight_file1))
        model2.load_state_dict(torch.load(pretrained_weight_file2))
        model3.load_state_dict(torch.load(pretrained_weight_file3))
        model4.load_state_dict(torch.load(pretrained_weight_file4))
        model5.load_state_dict(torch.load(pretrained_weight_file5))
        model6.load_state_dict(torch.load(pretrained_weight_file6))

        model1.cuda()
        model1.eval()
        model2.cuda()
        model2.eval()
        model3.cuda()
        model3.eval()
        model4.cuda()
        model4.eval()
        model5.cuda()
        model5.eval()
        model6.cuda()
        model6.eval()

        # load UAP generator and discriminator
        pth_path = '/home/airlab/Desktop/EEG/code/eeg_uap_airlab/result/hyper/'
        save_file_name = pth_path + param.dataset + '_net_condition%d.pth' % (fold)

        generator = GenResNetHyper(1, param.num_channel, param.num_length)
        generator.apply(weights_init)
        print('Generator weight initialized')
        generator.train()
        generator.cuda()

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        norm_type = param.norm_type
        norm_limit = param.epsilon

        embedding_size = 11

        for i in range(num_epoch):

            num_positive_nontargeted = 0
            num_total_nontargeted = 0
            num_positive_targeted = 0
            num_total_targeted = 0

            t0 = time.time()

            for train_x, train_y in train_loader:

                # -------------------------------------------------------
                # 0     :1 - eegnet     / 0: when using other model
                # 1     :1 - dconvnet   / 0: when using other model
                # 2     :1 - sconvnet   / 0: when using other model
                # 3     :1 - resnet     / 0: when using other model
                # 4     :1 - tidnet     / 0: when using other model
                # 5     :1 - vgg        / 0: when using other model
                # -------------------------------------------------------
                # 6    : 1 - non targeted
                # -------------------------------------------------------
                # 7    : 1 - when target class is 0
                # 8    : 1 - when target class is 1
                # 9    : 1 - when target class is 2
                # 10    : 1 - when target class is 3

                model_list = [model1, model2, model3, model4, model5, model6]
                model_list_copy = [model1, model2, model3, model4, model5, model6]
                flag = np.array([0 for i in range(len(model_list) * (param.num_class + 1))])
                while flag.sum() != len(model_list) * (param.num_class + 1):

                    model = random.choice(model_list)
                    model_idx = model_list.index(model)
                    attack_type_idx = random.choice([i for i in range(6, 6 + (param.num_class + 1))])

                    flag_index = (param.num_class + 1) * model_idx + ((attack_type_idx) - 6)

                    if flag[flag_index] == 1:
                        continue
                    else:
                        # Set embedding
                        embedding = np.zeros(embedding_size, dtype=np.float32)
                        embedding[model_idx] = 1.0
                        embedding[attack_type_idx] = 1.0
                        embedding_cuda = torch.from_numpy(embedding.reshape(embedding_size, 1, 1)).type(torch.FloatTensor).cuda()

                        if attack_type_idx == 6:
                            attack_type = 'non-targeted'
                            # Get original prediction from victim model
                            with torch.no_grad():
                                output = model.forward(train_x.cuda())
                                output_sm = F.softmax(output, dim=1)
                                _, target_label = torch.max(output_sm, 1)
                                target_label = target_label.cuda()
                        elif attack_type_idx == 7:
                            attack_type = 'targeted'
                            target_label = torch.mul(train_y, 0).cuda()
                        elif attack_type_idx == 8:
                            attack_type = 'targeted'
                            target_label = torch.add(torch.mul(train_y, 0), 1).cuda()
                        elif attack_type_idx == 9:
                            attack_type = 'targeted'
                            target_label = torch.add(torch.mul(train_y, 0), 2).cuda()
                        elif attack_type_idx == 10:
                            attack_type = 'targeted'
                            target_label = torch.add(torch.mul(train_y, 0), 3).cuda()

                        optimizer.zero_grad()
                        adv_exam_cuda = generator(train_x.cuda(), embedding_cuda)

                        # Scale
                        norm_exam = adv_exam_cuda.view(adv_exam_cuda.shape[0], -1)
                        if norm_type == 'inf':
                            norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                        elif norm_type == 'L2':
                            norm_exam = torch.norm(norm_exam, p=2, dim=1)
                        adv_exam_cuda = torch.mul(adv_exam_cuda / norm_exam.view(adv_exam_cuda.shape[0], 1, 1, 1), norm_limit)
                        train_x_adv = torch.add(train_x.cuda(), adv_exam_cuda)

                        # Do clamping per channel
                        for cii in range(param.num_channel):
                            train_x_adv[:, :, cii, :] = train_x_adv[:, :, cii, :].clone().clamp(
                                min=train_x[:, :, cii, :].min(), max=train_x[:, :, cii, :].max())

                        output = model.forward(train_x_adv)

                        if attack_type == 'non-targeted':
                            loss = torch.log(loss_func(1-F.softmax(output, dim=1), ))
                        elif attack_type == 'targeted':
                            loss = loss_func(output, target_label)

                        loss.backward()
                        optimizer.step()

                        # Non-target acc
                        output_sm = F.softmax(output, dim=1)
                        _, output_index = torch.max(output_sm, 1)
                        res_test = output_index.cpu().detach().numpy()

                        if attack_type == 'non-targeted':
                            tp_test = (res_test == train_y.cpu().detach().numpy()).sum()
                            num_positive_nontargeted = num_positive_nontargeted + tp_test
                            num_total_nontargeted = num_total_nontargeted + res_test.shape[0]
                        else:
                            tp_test = (res_test == target_label.cpu().detach().numpy()).sum()
                            num_positive_targeted = num_positive_targeted + tp_test
                            num_total_targeted = num_total_targeted + res_test.shape[0]

                        flag[flag_index] = 1

            #scheduler.step()
            t1 = time.time()
            test_accuracy_nontargeted = num_positive_nontargeted / num_total_nontargeted
            test_accuracy_targeted = num_positive_targeted / num_total_targeted
            print('Epoch:%d Train_loss:%.4f Time:%.4f Non-Target acc:%.4f (%d/%d) Target acc:%.4f (%d/%d)'%(i, loss.cpu().detach(), t1 - t0, test_accuracy_nontargeted, num_positive_nontargeted, num_total_nontargeted, test_accuracy_targeted,
                num_positive_targeted, num_total_targeted))

            # Save weights of generator
            if (i+1) % 5 == 0:
                torch.save(generator.state_dict(), save_file_name)
                print('Saved weight at' + save_file_name)

        print('Test info')
        generator.load_state_dict(torch.load(save_file_name))
        generator.eval()

        model_list = [model1, model2, model3, model4, model5, model6]
        model_name = ['eegnet', 'dconvnet', 'sconvnet', 'resnet', 'tidnet', 'vgg']

        for model_idx in range(len(model_list)):

            model = model_list[model_idx]

            num_positive_targeted = 0
            num_total_targeted = 0
            num_positive_nontargeted = 0
            num_total_nontargeted = 0
            num_fool_nt = 0
            num_fool_t = 0

            for attack_type_idx in range(6, 6 + (param.num_class + 1)):

                num_positive = 0
                num_total = 0

                # Set embedding for non-target attack
                embedding = np.zeros(embedding_size, dtype=np.float32)
                embedding[model_idx] = 1.0
                embedding[attack_type_idx] = 1.0

                embedding_cuda = torch.from_numpy(embedding.reshape(embedding_size, 1, 1)).type(torch.FloatTensor).cuda()

                for test_x, test_y in test_loader:

                    with torch.no_grad():
                        output = model.forward(test_x.cuda())
                        output_sm = F.softmax(output, dim=1)
                        _, original_prediction = torch.max(output_sm, 1)
                        res_test = original_prediction.cpu().detach().numpy()
                    tp_test = (res_test == test_y.cpu().detach().numpy()).sum()
                    num_positive = num_positive + tp_test
                    num_total = num_total + res_test.shape[0]

                    adv_exam_cuda = generator(test_x.cuda(), embedding_cuda)
                    norm_exam = adv_exam_cuda.view(adv_exam_cuda.shape[0], -1)
                    if norm_type == 'inf':
                        norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                    elif norm_type == 'L2':
                        norm_exam = torch.norm(norm_exam, p=2, dim=1)
                    adv_exam_cuda = torch.mul(adv_exam_cuda / norm_exam.view(adv_exam_cuda.shape[0], 1, 1, 1),
                                              norm_limit)

                    test_x_adv = torch.add(test_x.cuda(), adv_exam_cuda)

                    if attack_type_idx == 6:
                        target_label = test_y.cuda()
                    elif attack_type_idx == 7:
                        target_label = torch.add(torch.mul(test_y, 0), 0).cuda()
                    elif attack_type_idx == 8:
                        target_label = torch.add(torch.mul(test_y, 0), 1).cuda()
                    elif attack_type_idx == 9:
                        target_label = torch.add(torch.mul(test_y, 0), 2).cuda()
                    elif attack_type_idx == 10:
                        target_label = torch.add(torch.mul(test_y, 0), 3).cuda()

                    # Do clamping per channel
                    for cii in range(param.num_channel):
                        test_x_adv[:, :, cii, :] = test_x_adv[:, :, cii, :].clone().clamp(
                            min=test_x[:, :, cii, :].min(), max=test_x[:, :, cii, :].max())

                    with torch.no_grad():
                        output = model.forward(test_x_adv)
                        output_sm = F.softmax(output, dim=1)
                        _, output_index = torch.max(output_sm, 1)
                        res_test = output_index.cpu().detach().numpy()

                    tp_test = (res_test == target_label.cpu().detach().numpy()).sum()

                    if attack_type_idx == 6:
                        num_positive_nontargeted = num_positive_nontargeted + tp_test
                        num_total_nontargeted = num_total_nontargeted + res_test.shape[0]
                        num_fool_nt += ((original_prediction.cpu().detach().numpy()) != res_test).sum()
                    else:
                        num_positive_targeted = num_positive_targeted + tp_test
                        num_total_targeted = num_total_targeted + res_test.shape[0]
                        num_fool_t += ((original_prediction.cpu().detach().numpy()) != res_test).sum()


            test_accuracy = num_positive / num_total
            test_accuracy_nontargeted = num_positive_nontargeted / num_total_nontargeted
            fooling_rate_nt = num_fool_nt / num_total_nontargeted
            test_accuracy_targeted = num_positive_targeted / num_total_targeted
            fooling_rate_t = num_fool_t / num_total_targeted

            print('Model:', model_name[model_idx])
            print('Clean acc:%.4f Non-Target acc:%.4f Target acc:%.4f' % (test_accuracy, test_accuracy_nontargeted, test_accuracy_targeted))
            print('Nt Fooling ratio: %.4f (%d / %d)  T Fooling ratio: %.4f' %(fooling_rate_nt, num_fool_nt, num_total_nontargeted, fooling_rate_t))

if __name__ == '__main__':

    no_gpu = 0

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_ner2015_eegnet.cfg'

    print(conf_file_name)
    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d' % no_gpu)

    train(conf)
