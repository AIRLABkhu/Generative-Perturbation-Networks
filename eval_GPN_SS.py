#  20210713

import sys
import configparam
from torchsummary import summary
import numpy as np

from models import *
from adversarial_models import *
from dataloaders.amigos_cnn_loader import amigos_cnn_loader
from dataloaders.deap_cnn_loader import deap_cnn_loader
from dataloaders.physionet_cnn_loader import physionet_cnn_loader
from dataloaders.ner2015_cnn_loader import ner2015_cnn_loader
from dataloaders.data_split import data_split

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
    elif param.model == 'newnet':
        print('VGG')
        model = vgg_eeg(pretrained=False, num_classes=param.num_class)

        print(model)

    if param.use_pretrained == 0:
        print('use pretrained has to be 1')
        exit()

    pretrained_weight_file = param.result_path + '/pretrained/' + param.pretrained_name
    print(pretrained_weight_file)
    model.load_state_dict(torch.load(pretrained_weight_file))

    model.eval()
    model.cuda()

    # summary(model, (1, param.num_channel, param.num_length))

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

    split = data_split(data_set, param, False)
    train_loader, val_loader, test_loader = split.get_split(batch_size=batch_size, num_workers=12)

    num_positive = 0
    num_total = 0

    num_positive_uap_exam = 0
    num_total_uap_exam = 0

    idx = 0

    #  load pretrained parameter
    generator = GenResNet(1, param.num_channel, param.num_length)
    if param.attack_type == 'non-targeted':
        pretrained_weight_file_uap_net = param.uap_path + 'air_uap_net_nt.pth'
    elif param.attack_type == 'targeted':
        pretrained_weight_file_uap_net = param.uap_path + 'air_uap_net_t%d.pth' % param.attack_target
    print(pretrained_weight_file_uap_net)
    generator.load_state_dict(torch.load(pretrained_weight_file_uap_net))
    generator.eval()
    generator.cuda()
    # summary(generator, (1, param.num_channel, param.num_length))

    norm_type = param.norm_type
    norm_limit = param.epsilon
    l2 = 0.0
    for test_x, test_y in test_loader:

        output = model.forward(test_x.cuda())

        output_sm = F.softmax(output, dim=1)

        _, output_index = torch.max(output_sm, 1)
        res = output_index.cpu().detach().numpy()

        tp = (res == test_y.detach().numpy()).sum()

        num_positive = num_positive + tp
        num_total = num_total + res.shape[0]

        # Adversarial result
        if param.attack_type == 'non-targeted':
            target_label = test_y.cuda()
        elif param.attack_type == 'targeted':
            target_label = torch.add(torch.mul(test_y, 0), param.attack_target).cuda()

        adv_exam_cuda_test = generator(test_x.cuda())
        norm_exam = adv_exam_cuda_test.view(adv_exam_cuda_test.shape[0], -1)
        #print('L2 size of perturbation:', torch.mean(torch.norm(norm_exam, p=2)))
        l2 += torch.mean(torch.norm(norm_exam, p=2)).cpu().detach()

        if norm_type == 'inf':
            norm_exam = torch.norm(norm_exam, p=float('inf'), dim=1)
        elif norm_type == 'L2':
            norm_exam = torch.norm(norm_exam, p=2)

        adv_exam_cuda = torch.mul(adv_exam_cuda_test / norm_exam.view(adv_exam_cuda_test.shape[0], 1, 1, 1), norm_limit)
        test_x_adv = torch.add(test_x.cuda(), adv_exam_cuda)

        output_uap_exam = model.forward(test_x_adv)

        output_sm_uap_exam = F.softmax(output_uap_exam, dim=1)
        _, output_index_uap_exam = torch.max(output_sm_uap_exam, 1)
        res_uap_exam = output_index_uap_exam.cpu().detach().numpy()

        tp_uap_exam = (res_uap_exam == target_label.cpu().detach().numpy()).sum()

        num_positive_uap_exam = num_positive_uap_exam + tp_uap_exam
        num_total_uap_exam = num_total_uap_exam + res_uap_exam.shape[0]

    test_accuracy = num_positive / num_total

    print('test accuracy: %.4f ( %d / %d)'%(test_accuracy, num_positive, num_total))
    eval_result = np.array([test_accuracy, num_positive, num_total])
    np.savetxt(param.result_path + 'evaluation_result.txt', eval_result, fmt='%1.4f')

    print('---')

    test_accuracy_uap_exam = num_positive_uap_exam / num_total_uap_exam

    print('test accuracy UAP net: %.4f ( %d / %d)' % (test_accuracy_uap_exam, num_positive_uap_exam, num_total_uap_exam))
    eval_result_uap_exam = np.array([test_accuracy_uap_exam, num_positive_uap_exam, num_total_uap_exam])

    print('---')
    print('L2 size of perturbation:', l2/num_total)
if __name__ == '__main__':

    no_gpu = 3

    if len(sys.argv) > 1:
        conf_file_name = sys.argv[1]
        if len(sys.argv) > 2:
            no_gpu = int(sys.argv[2])
    else:
        conf_file_name = './config/non-target/eval_physionet_resnet.cfg'
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


