#  20210713

import sys
import configparam
from torchsummary import summary
import numpy as np

from models import *
from dataloaders.amigos_cnn_loader import amigos_cnn_loader
from dataloaders.deap_cnn_loader import deap_cnn_loader
from dataloaders.physionet_cnn_loader import physionet_cnn_loader
from dataloaders.ner2015_cnn_loader import ner2015_cnn_loader
from dataloaders.data_split import data_split

from sklearn.model_selection import KFold
from adversarial_models.GenResNetHyper import *

torch.manual_seed(0)
k_folds = 5

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

    if param.use_pretrained == 0:
        print('use pretrained has to be 1')
        exit()

    pretrained_weight_file = param.result_path + '/pretrained/' + param.pretrained_name
    print(pretrained_weight_file)
    model.load_state_dict(torch.load(pretrained_weight_file))

    model.eval()
    model.cuda()

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
    results = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):
        # Print
        print('-----------------------')
        print(f'FOLD {fold}')
        print('-----------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        # train_ids, val_ids = train_test_split(train_ids, test_size=0.25, shuffle=True, random_state=0)

        np.random.seed(0)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=param.batch_size, sampler=test_subsampler,
                                                  num_workers=12)

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

        # Constraint on magnitude of perturbation
        norm_limit = param.epsilon

        # generator = GenResNetHyper(1, param.num_channel, param.num_length)
        # pth_path = '/home/airlab/Desktop/EEG/code/eeg_uap_airlab/result/hyper/'
        # save_file_name = pth_path + param.model + '_hyper%d.pth' % (fold)
        # generator.load_state_dict(torch.load(save_file_name))  # If there's pretrained weight
        # generator.eval()
        # generator.cuda()

        # Reset for test
        clean_num_positive = 0
        clean_num_total = 0
        num_positive = 0
        num_total = 0
        num_fool = 0

        if param.attack_type == 'non-targeted':
            # uap_file_name = param.uap_path + 'uap_air_exam_nt_fold%d.npy'%fold # GUP
            uap_file_name = param.uap_path + '_uap_tlm_non_targeted_fold%d.npy'%fold # TLM
        else:
            uap_file_name = param.uap_path + 'uap_air_exam_t%d_fold%d.npy' % (param.attack_target, fold)

        for test_x, test_y in test_loader:
            adv_exam_cuda_perturbation = np.load(uap_file_name)
            print('Load perturbation at' + uap_file_name)
            adv_exam_cuda_perturbation = torch.from_numpy(adv_exam_cuda_perturbation).cuda()
            adv_exam_cuda_perturbation = adv_exam_cuda_perturbation.cuda()

            #  Generate perturbation
            # adv_exam_cuda = generator(init_noise_cuda, embedding_cuda)
            # norm_exam = 1
            # if norm_type == 'inf':
            #     norm_exam = torch.norm(adv_exam_cuda, p=float('inf'))
            # elif norm_type == 'L2':
            #     norm_exam = torch.norm(adv_exam_cuda, p=2)

            # print('norm: %f'%norm_exam)
            # adv_exam_cuda_perturbation = adv_exam_cuda * (norm_limit / norm_exam)

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

        print('Adversarial test result on fold {}: {:.4f} -> {:.4f}, test fooling ratio {:.4f}'.format(fold,
                                                                                                       clean_test_accuracy,
                                                                                                       test_accuracy,
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

if __name__ == '__main__':

    no_gpu = 0

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_ner2015_sconvnet.cfg'
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


