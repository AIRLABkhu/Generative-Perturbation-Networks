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
from torch.utils.data.sampler import SubsetRandomSampler
import random

from models import *
from adversarial_models.GenResNetMulti import *
from lost_functions import *
from dataloaders.amigos_cnn_loader_subj import amigos_cnn_loader
from dataloaders.deap_cnn_loader_subj import deap_cnn_loader
from dataloaders.physionet_cnn_loader_subj import physionet_cnn_loader
from dataloaders.ner2015_cnn_loader_subj import ner2015_cnn_loader
from dataloaders.data_split import data_split

from sklearn.model_selection import KFold

k_folds = 5
torch.manual_seed(0)
random.seed(0)

def train(param):
    param.PrintConfig()
    learning_rate = param.learning_rate
    batch_size = param.batch_size
    num_epoch = param.num_epoch

    res_list_test = np.array([]).reshape((0, 3))

    # Set Model
    model1 = EEGNet(param.num_channel, param.num_length, param.num_class)
    model2 = DeepConvNet(param.num_channel, param.num_length, param.num_class)
    model3 = ShallowConvNet(param.num_channel, param.num_length, param.num_class)
    model4 = ResNet8(param.num_class)
    model5 = TIDNet(in_chans=param.num_channel, n_classes=param.num_class, input_window_samples=param.num_length)
    model6 = vgg_eeg(pretrained=False, num_classes=param.num_class)

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

        # Sample elements randomly from a given list of ids, no replacement.
        print('train ids:', train_ids)
        print('test ids:', test_ids)

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

        if param.use_predefined_idx == 0:
            print('pretrained index has to be 1')
            exit()

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True,num_workers=12)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=12)

        # Load pre-trained wight
        if param.dataset == 'ner2015':
            pretrained_weight_file1 = './result/' + param.dataset + '_eegnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_eegnet_e0050_subj.pth'
            pretrained_weight_file2 = './result/' + param.dataset + '_dconvnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_dconvnet_e0050_subj.pth'
            pretrained_weight_file3 = './result/' + param.dataset + '_sconvnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_sconvnet_e0050_subj.pth'
            pretrained_weight_file4 = './result/' + param.dataset + '_resnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_resnet_e0050_subj.pth'
            pretrained_weight_file5 = './result/' + param.dataset + '_tidnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_tidnet_e0050_subj.pth'
            pretrained_weight_file6 = './result/' + param.dataset + '_vgg' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_vgg_e0050_subj.pth'
        else:
            pretrained_weight_file1 = './result/' + param.dataset + '_eegnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_eegnet_e0200_subj.pth'
            pretrained_weight_file2 = './result/' + param.dataset + '_dconvnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_dconvnet_e0200_subj.pth'
            pretrained_weight_file3 = './result/' + param.dataset + '_sconvnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_sconvnet_e0200_subj.pth'
            pretrained_weight_file4 = './result/' + param.dataset + '_resnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_resnet_e0200_subj.pth'
            pretrained_weight_file5 = './result/' + param.dataset + '_tidnet' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_tidnet_e0200_subj.pth'
            pretrained_weight_file6 = './result/' + param.dataset + '_vgg' + '//pretrained/' + f'fold{fold}_' + param.dataset + '_vgg_e0200_subj.pth'

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
        pth_path = '/home/hj/PycharmProjects/EEG/eeg_uap_airlab/result/hyper/'
        save_file_name = pth_path + param.dataset + '_multi%d_subj.pth' % (fold)

        # load UAP generator and discriminator
        generator = GenResNetMulti(1, param.num_channel, param.num_length, 6*(param.num_class+1))
        generator.train()
        generator.cuda()

        loss_func = nn.CrossEntropyLoss()
        # loss_func = FocalLoss()

        optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

        norm_type = param.norm_type
        norm_limit = param.epsilon

        loss_total = 0.0
        '''
        for i in range(num_epoch):
            loss_epoch = 0.0
            cnt_epoch = 0

            num_positive_nt = 0
            num_total_nt = 0
            num_positive_t = 0
            num_total_t = 0

            t0 = time.time()

            for train_x, train_y in train_loader:

                train_x = train_x.cuda()
                model_list = [model1, model2, model3, model4, model5, model6]
                model_list_copy = [model1, model2, model3, model4, model5, model6]

                while len(model_list) != 0:

                    optimizer.zero_grad()

                    model = random.choice(model_list)
                    model_idx = model_list_copy.index(model)
                    model_list.remove(model)
                    # print(model_idx)

                    # Generate perturbations in this order: non-traget, target 0, target 1, target 2, ...
                    adv_exam_multi_cuda = generator(train_x)
                    # print('adv_exam_multi_cuda shape:', adv_exam_multi_cuda.shape)

                    loss_multi = 0

                    # Get loss per each non-target/target attack!
                    for m in range(param.num_class+1):

                        idx = (param.num_class + 1) * model_idx + m
                        adv_exam_cuda = adv_exam_multi_cuda[:, idx:idx + 1, :, :]
                        # print('adv_exam_cuda shape:', adv_exam_cuda.shape)

                        norm_exam = adv_exam_cuda.view(adv_exam_cuda.shape[0], -1)
                        if norm_type == 'inf':
                            norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                        elif norm_type == 'L2':
                            norm_exam = torch.norm(norm_exam, p=2)

                        adv_exam_cuda = torch.mul(adv_exam_cuda / norm_exam.view(adv_exam_cuda.shape[0], 1, 1, 1), norm_limit)

                        train_x_adv = torch.add(train_x, adv_exam_cuda)

                        # Do clamping per channel
                        for cii in range(param.num_channel):
                            train_x_adv[:, :, cii, :] = train_x_adv[:, :, cii, :].clone().clamp(min=train_x[:, :, cii, :].min(),
                                                                                                max=train_x[:, :, cii, :].max())
                        # Feed Generator
                        output = model.forward(train_x_adv)

                        if m == 0: # For non-target loss
                            target_label = train_y.cuda()
                            loss = loss_func(1-F.softmax(output, dim=1), target_label)
                            loss_multi += loss
                        else: # For target loss
                            target_label = torch.add(torch.mul(train_y, 0), m-1)
                            loss = loss_func(output, target_label.cuda())
                            loss_multi += ((1.0) / param.num_class * loss)

                        output_sm = F.softmax(output, dim=1)
                        _, output_index = torch.max(output_sm, 1)
                        res = output_index.cpu().detach().numpy()

                        if m == 0:
                            tp = (res == train_y.detach().numpy()).sum()
                            num_positive_nt = num_positive_nt + tp
                            num_total_nt = num_total_nt + res.shape[0]
                        else:
                            tp = (res == target_label.cpu().detach().numpy()).sum()
                            num_positive_t = num_positive_t + tp
                            num_total_t = num_total_t + res.shape[0]

                    loss_multi.backward()
                    optimizer.step()

                    loss_epoch = loss_epoch + loss_multi
                    cnt_epoch = cnt_epoch + 1

            train_accuracy_nt = num_positive_nt / num_total_nt
            train_accuracy_t = num_positive_t / num_total_t

            loss_total = loss_total + (loss_epoch.cpu().detach().numpy() / cnt_epoch)

            # save Weights
            if (i+1) % 5 == 0:
                torch.save(generator.state_dict(), save_file_name)
                print('Saved at' + save_file_name)

            t1 = time.time()
            print(
                'epoch:{} loss:{:.4f} loss_avg:{:.4f} train accuracy_nt:{:.4f} train accuracy_t:{:.4f} time:{:.4f}'.format(
                    i + 1, (loss_epoch / cnt_epoch), (loss_total / (i + 1)), train_accuracy_nt, train_accuracy_t, (t1 - t0)))
        '''
        # Test
        num_positive_test_nt = 0
        num_total_test_nt = 0

        num_positive_test_t = 0
        num_total_test_t = 0

        print('Test info')
        generator.load_state_dict(torch.load(save_file_name))
        generator.eval()

        model_list = [model1, model2, model3, model4, model5, model6]
        model_name = ['eegnet', 'dconvnet', 'sconvnet', 'resnet', 'tidnet', 'vgg']

        for model_idx in range(len(model_list)):

            generator.eval()
            model = model_list[model_idx]

            num_positive_targeted = 0
            num_total_targeted = 0
            num_positive = 0
            num_total = 0
            num_positive_nontargeted = 0
            num_total_nontargeted = 0
            num_fool_nt = 0
            num_fool_t = 0

            for test_x, test_y in test_loader:

                test_x = test_x.cuda()
                adv_exam_multi_cuda = generator(test_x)

                # Clean Acc
                with torch.no_grad():
                    output = model.forward(test_x.cuda())
                    output_sm = F.softmax(output, dim=1)
                    _, original_prediction = torch.max(output_sm, 1)
                    res_test = original_prediction.cpu().detach().numpy()
                tp_test = (res_test == test_y.cpu().detach().numpy()).sum()
                num_positive = num_positive + tp_test
                num_total = num_total + res_test.shape[0]

                # Get loss per each non-target/target attack!
                for m in range(param.num_class + 1):

                    # Set index of perturbation accoridng to model and attack type
                    idx = (param.num_class + 1) * model_idx + m
                    adv_exam_cuda = adv_exam_multi_cuda[:, idx:idx + 1, :, :]

                    norm_exam = adv_exam_cuda.view(adv_exam_cuda.shape[0], -1)
                    if norm_type == 'inf':
                        norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
                    elif norm_type == 'L2':
                        norm_exam = torch.norm(norm_exam, p=2)

                    adv_exam_cuda = torch.mul(adv_exam_cuda / norm_exam.view(adv_exam_cuda.shape[0], 1, 1, 1),
                                              norm_limit)

                    test_x_adv = torch.add(test_x, adv_exam_cuda)

                    # Do clamping per channel
                    for cii in range(param.num_channel):
                        test_x_adv[:, :, cii, :] = test_x_adv[:, :, cii, :].clone().clamp(
                            min=test_x[:, :, cii, :].min(),
                            max=test_x[:, :, cii, :].max())

                    # Feed Generator
                    with torch.no_grad():
                        output = model.forward(test_x_adv)
                        output_sm = F.softmax(output, dim=1)
                        _, output_index = torch.max(output_sm, 1)
                        res_test = output_index.cpu().detach().numpy()

                    if m == 0:  # For non-target loss
                        target_label = test_y.cuda()
                        tp_test = (res_test == target_label.cpu().detach().numpy()).sum()
                        num_positive_nontargeted = num_positive_nontargeted + tp_test
                        num_total_nontargeted = num_total_nontargeted + res_test.shape[0]

                        # Calculate fooling rate
                        num_fool_nt += ((original_prediction.cpu().detach().numpy()) != res_test).sum()

                    else:  # For target loss
                        target_label = torch.add(torch.mul(test_y, 0), m - 1)
                        tp_test = (res_test == target_label.cpu().detach().numpy()).sum()
                        num_positive_targeted = num_positive_targeted + tp_test
                        num_total_targeted = num_total_targeted + res_test.shape[0]

                        # Calculate fooling rate
                        num_fool_t += ((original_prediction.cpu().detach().numpy()) != res_test).sum()

            test_accuracy = num_positive / num_total
            test_accuracy_nontargeted = num_positive_nontargeted / num_total_nontargeted
            test_fooling_nontargeted = num_fool_nt / num_total_nontargeted
            test_accuracy_targeted = num_positive_targeted / num_total_targeted
            test_fooling_targeted = num_fool_t / num_total_targeted

            print('Model:', model_name[model_idx])
            print('Clean acc:%.4f Non-Target acc:%.4f Target acc:%.4f' % (test_accuracy, test_accuracy_nontargeted, test_accuracy_targeted))
            print('Nt Fooling ratio: %.4f (%d / %d)  T Fooling ratio: %.4f' %(test_fooling_nontargeted, num_fool_nt, num_total_nontargeted, test_fooling_targeted))


if __name__ == '__main__':

    no_gpu = 1

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_deap_eegnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d' % no_gpu)

    train(conf)
