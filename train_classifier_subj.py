# Import library
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import sys
import configparam
import time

# Import pretrained victim models
from models import *

# Import DataLoaders
from dataloaders.amigos_cnn_loader_subj import amigos_cnn_loader
from dataloaders.deap_cnn_loader_subj import deap_cnn_loader
from dataloaders.physionet_cnn_loader_subj import physionet_cnn_loader
from dataloaders.ner2015_cnn_loader_subj import ner2015_cnn_loader

# K-folds validation
from sklearn.model_selection import KFold
k_folds = 5
torch.manual_seed(0)
np.random.seed(0)

def train(param):

    # Define Hyper-parameters
    param.PrintConfig()
    learning_rate = param.learning_rate
    num_epoch = param.num_epoch
    batch_size = param.batch_size

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    subject_list = [i for i in range(param.num_subject)]

    for fold, (train_ids, test_ids) in enumerate(kfold.split(subject_list)):

        res_list_test = np.array([]).reshape((0, 3))

        # Print fold info
        print('-----------------------')
        print(f'FOLD {fold}')
        print('-----------------------')

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

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        # set model
        if param.model == 'eegnet':
            print('Model: EEGNet')
            model = EEGNet(param.num_channel, param.num_length, param.num_class)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        elif param.model == 'sconvnet':
            print('Shallow Conv Net')
            model = ShallowConvNet(param.num_channel, param.num_length, param.num_class)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)
        elif param.model == 'dconvnet':
            print('Deep Conv Net')
            model = DeepConvNet(param.num_channel, param.num_length, param.num_class)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)
        elif param.model == 'resnet':
            print('ResNet')
            model = ResNet8(param.num_class)
        elif param.model == 'tidnet':
            print('TIDNet')
            model = TIDNet(in_chans=param.num_channel, n_classes=param.num_class, input_window_samples=param.num_length)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        elif param.model == 'vgg':
            print('VGG')
            model = vgg_eeg(pretrained=False, num_classes=param.num_class)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

        model.train()
        model.cuda()

        loss_total = 0.0

        # Define optimizer and scheduler
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        for i in range(num_epoch):
            loss_epoch = 0.0
            cnt_epoch = 0

            num_positive = 0
            num_total = 0

            t0 = time.time()

            for train_x, train_y in train_loader:

                train_x = train_x.cuda()
                train_y = train_y.cuda()

                # Signal augmentation by adding gaussian noises, then clip into proper range
                sigma = 0.01
                add_noise = torch.normal(0, sigma, (train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3]))
                train_x = torch.clamp(train_x + add_noise.cuda(), min=0.0, max=1.0)

                optimizer.zero_grad()
                output = model.forward(train_x.cuda())
                loss = loss_func(output, train_y)

                loss.backward()
                optimizer.step()

                output_sm = F.softmax(output, dim=1)
                _, output_index = torch.max(output_sm, 1)
                res = output_index.cpu().detach().numpy()

                tp = (res == train_y.cpu().detach().numpy()).sum()

                num_positive = num_positive + tp
                num_total = num_total + res.shape[0]

                loss_epoch = loss_epoch + loss.detach()
                cnt_epoch = cnt_epoch + 1

            scheduler.step()
            train_accuracy = num_positive / num_total
            loss_total = loss_total + (loss_epoch / cnt_epoch)

            num_positive = 0
            num_total = 0

            for test_x, test_y in test_loader:
                test_x = test_x.cuda()
                test_y = test_y.cuda()

                with torch.no_grad():
                    output = model.forward(test_x)
                    output_sm = F.softmax(output, dim=1)
                    _, output_index = torch.max(output_sm, 1)
                    res = output_index.cpu().detach().numpy()
                    tp = (res == test_y.cpu().detach().numpy()).sum()

                num_positive = num_positive + tp
                num_total = num_total + res.shape[0]

            test_accuracy = num_positive / num_total


            t1 = time.time()

            print(
                'epoch:{} train loss:{:.4f} loss_avg:{:.4f} train accuracy:{:.4f} test accuracy:{:.4f} time:{:.4f}'.format(
                    i + 1, (loss_epoch / cnt_epoch), (loss_total/(i+1)), train_accuracy, test_accuracy, (t1 - t0),
                    ))

            # # Save result
            # res_list_test = np.append(res_list_test, np.array([[i + 1, train_accuracy,test_accuracy]]), axis=0)
            # np.savetxt(param.result_path + f'_{fold}_train_subj_result.txt', res_list_test, fmt='%1.4f')
            #
            # # Save models with 5 epochs intervals
            # if i != 0 and (i + 1) % 5 == 0:
            #     save_file_name = param.weight_path + f'fold{fold}_' + param.weight_prefix + '_e{:04d}_subj.pth'.format(i + 1)
            #     #save_file_name = param.weight_path + f'fold{fold}_' + param.weight_prefix + '.pth'
            #     #save_file_name = param.weight_path + param.weight_prefix + '_e{:04d}.pth'.format(i + 1)
            #     torch.save(model.state_dict(), save_file_name)
            #     print('saved at' + save_file_name)


if __name__ == '__main__':

    no_gpu = 0

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/train_ner2015_eegnet.cfg'
        # conf_file_name = './config/train_amigos_sconvnet.cfg'
        # conf_file_name = './config/train_deap_tidnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d'%no_gpu)

    train(conf)


