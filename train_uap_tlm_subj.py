'''
Total Loss Minimization(TLM) based UAP Generation in PyTorch

Reference:
[1] Zihan Liu, Lubin Meng, Xiao Zhang, Weili Fang and Dongrui Wu
    Universal adversarial perturbations for CNN classifiers in EEG-based BCIs. Journal Of Neural Engineering, 2021

Original implementation: https://github.com/ZihanLiu95/UAP_EEG
'''
# Import library
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import os
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
from sklearn.model_selection import KFold, train_test_split
k_folds = 5

torch.manual_seed(0)

def train(param):

    # Define Hyper-parmeters
    param.PrintConfig()
    learning_rate = param.learning_rate
    batch_size = param.batch_size
    num_epoch = param.num_epoch
    norm_limit = param.epsilon
    alpha = 0 # Coefficient for regularization

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # For fold results
    results = []

    subject_list = [i for i in range(param.num_subject)]
    for fold, (train_ids, test_ids) in enumerate(kfold.split(subject_list)):

        # Print fold info
        print('-----------------------')
        print(f'FOLD {fold}')
        print('-----------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_ids, val_ids = train_test_split(train_ids, test_size=0.25, shuffle=True, random_state=0)

        # Load dataset!
        if param.dataset == 'amigos':
            train_dataset = amigos_cnn_loader(param, subject_list=train_ids)
            val_dataset = amigos_cnn_loader(param, subject_list=val_ids)
            test_dataset = amigos_cnn_loader(param, subject_list=test_ids)
        elif param.dataset == 'deap':
            train_dataset = deap_cnn_loader(param, subject_list=train_ids)
            val_dataset = deap_cnn_loader(param, subject_list=val_ids)
            test_dataset = deap_cnn_loader(param, subject_list=test_ids)
        elif param.dataset == 'physionet':
            train_dataset = physionet_cnn_loader(param, subject_list=train_ids)
            val_dataset = physionet_cnn_loader(param, subject_list=val_ids)
            test_dataset = physionet_cnn_loader(param, subject_list=test_ids)
        elif param.dataset == 'ner2015':
            train_dataset = ner2015_cnn_loader(param, subject_list=train_ids)
            val_dataset = ner2015_cnn_loader(param, subject_list=val_ids)
            test_dataset = ner2015_cnn_loader(param, subject_list=test_ids)

        # Define dataloaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        # set victim model
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
        elif param.model == 'tidnet':
            print('TIDNet')
            model = TIDNet(in_chans=param.num_channel, n_classes=param.num_class, input_window_samples=param.num_length)
        elif param.model == 'vgg':
            print('VGG')
            model = vgg_eeg(pretrained=False, num_classes=param.num_class)

        # Load pretrained weight for victim model
        pretrained_weight_file = param.result_path + '/pretrained/' + f'fold{fold}_' + param.pretrained_name
        print(pretrained_weight_file)
        model.load_state_dict(torch.load(pretrained_weight_file))

        model.eval()
        model.cuda()

        # Generate a single UAP
        np.random.seed(0)
        init_universal_noise = np.zeros((param.num_channel, param.num_length))
        init_noise = np.reshape(init_universal_noise, (1, param.num_channel, param.num_length))
        init_noise = init_noise[np.newaxis, :, :, :]
        init_universal_noise = torch.from_numpy(init_noise).type(torch.FloatTensor).cuda()
        init_universal_noise.requires_grad = True

        # Define loss criterion
        loss_func = nn.CrossEntropyLoss()

        # Define optimizer and scheduler
        optimizer = optim.Adam([init_universal_noise], lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) # Not used actually

        # Print attack type!
        print('Init perturbation shape :', init_universal_noise.size())
        if param.attack_type == 'non-targeted':
            print('Non-targeted Attack!')
        elif param.attack_type == 'targeted':
            print('%d class Targeted Attack!' % param.attack_target)

        # Set initial best acc
        if param.attack_type == 'non-targeted':
            best_val_accuracy = 100
        elif param.attack_type == 'targeted':
            best_val_accuracy = 0

        loss_total = 0

        for i in range(num_epoch):

            num_correct = 0
            num_total = 0

            loss_epoch = 0.0
            cnt_epoch = 0
            start_time = time.time()

            # Optimize UAP
            for train_x, train_y in train_loader:

                # Set labels depend on attack_type
                if param.attack_type == 'targeted':
                    target_label = torch.add(torch.mul(train_y, 0), param.attack_target)
                    target_label = target_label.cuda()
                    cur_noise_cuda = init_universal_noise
                elif param.attack_type == 'non-targeted':
                    target_label = train_y.cuda()
                    cur_noise_cuda = init_universal_noise
                adv_exam_cuda = cur_noise_cuda

                optimizer.zero_grad()

                # Add perturbation to input
                train_x_adv = torch.add(train_x.cuda(), adv_exam_cuda)

                if param.attack_type == 'non-targeted':
                    output = model.forward(train_x_adv)
                    loss = -loss_func(output, target_label)
                elif param.attack_type == 'targeted':
                    output = model.forward(train_x_adv)
                    loss = loss_func(output, target_label)

                # Constraint
                if param.norm_type == 'inf':
                    loss += 0
                elif param.norm_type == 'L2':
                    loss += alpha * (torch.mean(torch.square(adv_exam_cuda)))
                elif param.norm_type == 'L1':
                    loss += alpha * (torch.mean(torch.abs(adv_exam_cuda)))

                loss.backward()
                optimizer.step()

                # Clip perturbation according to norm_limit
                adv_exam_cuda = torch.clip(adv_exam_cuda, min=-norm_limit, max=norm_limit).cuda()

                # Calculate Train acc
                train_x_adv = torch.add(train_x.cuda(), adv_exam_cuda)
                output = model.forward(train_x_adv)
                _, output_index = torch.max(output, 1)
                res = output_index.cpu().detach().numpy()
                if param.attack_type == 'non-targeted':
                    tp = (res == train_y.detach().numpy()).sum()
                elif param.attack_type == 'targeted':
                    tp = (res == target_label.cpu().detach().numpy()).sum()
                num_correct += tp
                num_total += res.shape[0]
                loss_epoch = loss_epoch + loss.detach()
                cnt_epoch = cnt_epoch + 1

            scheduler.step()

            train_accuracy = num_correct / num_total

            num_correct = 0
            num_total = 0

            loss_total = loss_total + (loss_epoch / cnt_epoch)

            # validation -> Find best perturbation v
            for val_x, val_y in val_loader:

                # Add perturbation
                val_x = torch.add(val_x.cuda(), adv_exam_cuda).cuda()

                if param.attack_type == 'non-targeted':
                    val_target_label = val_y.cuda()
                elif param.attack_type == 'targeted':
                    val_target_label = torch.add(torch.mul(val_y, 0), param.attack_target).cuda()

                with torch.no_grad():
                    output = model.forward(val_x)
                    output_sm = F.softmax(output, dim=1)
                    _, output_index = torch.max(output_sm, 1)
                    res_test = output_index.cpu().detach().numpy()

                tp_test = (res_test == val_target_label.cpu().detach().numpy()).sum()

                num_correct = num_correct + tp_test
                num_total = num_total + len(val_y)

            val_accuracy = num_correct / num_total

            print('epoch:{} loss:{:.4f} time:{:.4f} lr:{} train_accuracy:{:.4f} val_accuracy:{:.4f}'.
                  format(i + 1, loss.detach(), time.time() - start_time, scheduler.get_last_lr(), train_accuracy,
                         val_accuracy))

            # Saved best universal perturbation
            if param.attack_type == 'non-targeted':
                if val_accuracy < best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    uap_exam = adv_exam_cuda.cpu().detach().numpy()[0, :, :, :]
                    uap_file_name = param.uap_path + '_uap_tlm_non_targeted_fold%d_subj.npy' % fold
                    np.save(uap_file_name, uap_exam)
                    print(uap_file_name, 'Saved best perturbation!')
            else:
                if val_accuracy > best_val_accuracy: # If attack_type is targeted, largest acc means that attack is successful!
                    best_val_accuracy = val_accuracy
                    uap_exam = adv_exam_cuda.cpu().detach().numpy()[0, :, :, :]
                    # save perturbation
                    uap_file_name = param.uap_path + '_uap_tlm_%d_targeted_fold%d_subj.npy' % (param.attack_target, fold)
                    np.save(uap_file_name, uap_exam)
                    print(uap_file_name, 'Saved best perturbation!')

        # Reset for test
        clean_num_positive = 0
        clean_num_total = 0
        num_positive = 0
        num_total = 0
        num_fool = 0

        for test_x, test_y in test_loader:

            # Load TLM-UAP
            adv_exam_cuda_perturbation = np.load(uap_file_name)
            adv_exam_cuda_perturbation = torch.from_numpy(adv_exam_cuda_perturbation).cuda()
            test_x_adv = torch.add(test_x.cuda(), adv_exam_cuda_perturbation)

            # Set label
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
        adv_test_accuracy = num_positive / num_total
        test_fooling_ratio = num_fool / num_total

        results.append([clean_test_accuracy, adv_test_accuracy, test_fooling_ratio])

        print('Adversarial test result on fold {}: {:.4f} -> {:.4f}, test fooling ratio {:.4f}'.format(fold,
                                                                                                       clean_test_accuracy,
                                                                                                       adv_test_accuracy,
                                                                                                       test_fooling_ratio))

    # Print fold results
    print(f'Finished K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_clean = 0.0
    sum_adv = 0.0
    sum_fool = 0.0
    for i in range(len(results)):
        print(
            'Fold : {}, test_acc : {:.4f} -> {:.4f}, test fooling ratio {:.4f}'.format(i, results[i][0], results[i][1],
                                                                                       results[i][2]))
        sum_clean += results[i][0]
        sum_adv += results[i][1]
        sum_fool += results[i][2]
    print('Average: {:.4f} -> {:.4f}, fooling ratio {:.4f}'.format(sum_clean / len(results), sum_adv / len(results),
                                                                   sum_fool / len(results)))
    # Save result
    result_list = np.array(results)
    result_list = np.append(result_list,
                            np.array([[sum_clean / len(results), sum_adv / len(results), sum_fool / len(results)]]),
                            axis=0)

    if param.attack_type == 'targeted':
        np.savetxt(param.uap_path + '_tlm_result_target%d_fold_subj.txt' % param.attack_target, result_list, fmt='%1.4f')
        print('saved at' + param.uap_path + '_tlm_result_target%d_fold_subj.txt' % param.attack_target)
    elif param.attack_type == 'non-targeted':
        np.savetxt(param.uap_path + '_tlm_result_non_target_fold_subj.txt', result_list, fmt='%1.4f')
        print('saved at' + param.uap_path + '_tlm_result_non_target_fold_subj.txt')

if __name__ == '__main__':

    no_gpu = 7

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_ner2015_eegnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d' % no_gpu)

    train(conf)


