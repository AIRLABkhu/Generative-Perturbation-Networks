'''
Deepfool based UAP Generation in PyTorch

Reference:
[1] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, Pascal Frossard
    Universal adversarial perturbations. CVPR, 2017
'''

#  Import library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

# Import ART(Adversarial Robustness Toolbox) library for DeepFool-UAP
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import UniversalPerturbation

# K-folds validation
from sklearn.model_selection import KFold
k_folds = 5

def train(param):

    # Define Hyper-parmeters
    param.PrintConfig()
    learning_rate = param.learning_rate
    batch_size = param.batch_size
    num_epoch = param.num_epoch
    res_list_test = np.array([]).reshape((0, 3))

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # For results per fold
    results = []

    subject_list = [i for i in range(param.num_subject)]

    for fold, (train_ids, test_ids) in enumerate(kfold.split(subject_list)):

        # Print fold info
        print('-----------------------')
        print(f'FOLD {fold}')
        print('-----------------------')

        print('train ids:', train_ids)
        print('test ids:', test_ids)

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

        x_train = []
        y_train = []
        x_test = []
        y_test = []

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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, num_workers=12)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=12)

        for data, labels in train_loader:
            for eeg in data.numpy():
                x_train.append(eeg)
            for label in labels.numpy():
                y_train.append(label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # print('x_train shape:', x_train.shape)
        # print('y_train shape:', y_train.shape)

        for data, labels in test_loader:
            for eeg in data.numpy():
                x_test.append(eeg)
            for label in labels.numpy():
                y_test.append(label)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # print('x_test shape:', x_test.shape)
        # print('y_test shape:', y_test.shape)

        # Define Criterion and Optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Create the ART classifier
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=loss_func,
            optimizer=optimizer,
            input_shape=(1, param.num_channel, param.num_length),
            nb_classes=param.num_class
        )

        # Generate adversarial train examples & perturbation
        start_time = time.time()
        '''
        attack = UniversalPerturbation(classifier=classifier, eps=param.epsilon, max_iter=5, delta=0.2,
                                       batch_size=batch_size, norm='inf')
        adv_x_train = attack.generate(x_train)
        adv_perturbation = attack.noise
        end_time = time.time()
        print('Generated perturbation in %.4f seconds!'%(end_time-start_time))
        '''
        # Clean accuracy
        predictions = classifier.predict(x_test)
        clean_acc = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)

        # Adversarial accuracy
        adv_perturbation = np.load(param.uap_path + 'df_uap_fold%d_subj.npy'%fold) # Load saved perturbation
        adv_predictions = classifier.predict(x_test + adv_perturbation)
        perturbated_acc = np.sum(np.argmax(adv_predictions, axis=1) == y_test) / len(y_test)

        # Fooling ratio
        fooling_ratio = np.sum(np.argmax(predictions, axis=1) != np.argmax(adv_predictions, axis=1)) / len(y_test)

        print('Clean Accuracy: %.4f'%clean_acc)
        print('Adversarial Accuracy: %.4f'%perturbated_acc)
        print('Fooling ration: %.4f'%fooling_ratio)

        results.append([clean_acc, perturbated_acc, fooling_ratio])
        print('Adversarial test result on fold {}: {:.4f} -> {:.4f}, test fooling ratio {:.4f}'.format(fold,
                                                                                                       clean_acc,
                                                                                                       perturbated_acc,
                                                                                                       fooling_ratio))
        # Save Universal perturbation per fold
        # np.save(param.uap_path + 'df_uap_fold%d_subj.npy'%fold, attack.noise)

    # Print fold results
    print(f'Finished K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_clean = 0.0
    sum_adv = 0.0
    sum_fool = 0.0
    for i in range(len(results)):
        print('Fold : {}, test_acc : {:.4f} -> {:.4f}, test fooling ratio {:.4f}'.format(i, results[i][0],
                                                                                         results[i][1],
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
        np.savetxt(param.uap_path + '_df_result_target%d_fold_subj.txt' % param.attack_target, result_list, fmt='%1.4f')
        print('saved at' + param.uap_path + '_df_result_non_target_fold%d_fold_subj.txt' % param.attack_target)
    elif param.attack_type == 'non-targeted':
        np.savetxt(param.uap_path + '_df_result_non_target_fold_subj.txt', result_list, fmt='%1.4f')
        print('saved at' + param.uap_path + '_df_result_non_target_fold_subj.txt')

if __name__ == '__main__':
    no_gpu = 0
    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_deap_eegnet.cfg'

    conf = configparam.ConfigParam()
    conf.LoadConfiguration(conf_file_name)

    torch.cuda.set_device(no_gpu)
    print('GPU allocation ID: %d'%no_gpu)

    train(conf)


