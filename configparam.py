import os

class ConfigParam:

    data_path = ''
    weight_path = ''
    weight_prefix = ''
    result_path = ''
    uap_path = ''
    use_pretrained = False
    use_predefined_idx = False
    pretrained_name = ''
    predefined_dataset_path = ''
    train_ratio = 0.0
    split_path = ''
    target_label = 'valence'

    num_subject = 0
    num_trial = 0
    num_channel = 0
    num_length = 0
    num_class = 0
    dataset_type = ''
    target_subject = []
    target_trial = []
    train_ratio = 0
    exception_subject = []
    exception_trial = []

    dataset = ''
    model = ''
    learning_ratio = 0
    num_epoch = 0
    batch_size = 0

    def __init__(self):
        return

    def LoadConfiguration(self, run_path):

        f = open(run_path, 'r')

        # print('- Load configuration -')
        while True:
            line = f.readline()
            if not line: break
            words = line.replace('\n','').split(': ')
            if len(words) > 1:
                if words[0] == 'data_path':
                    self.data_path = words[1]
                elif words[0] == 'result_path':
                    self.result_path = words[1]
                    self.weight_path = self.result_path + 'pretrained/'
                    self.split_path = self.result_path + 'split/'
                    self.uap_path = self.result_path + 'uap/'
                elif words[0] == 'weight_prefix':
                    self.weight_prefix = words[1]
                elif words[0] == 'use_pretrained':
                    self.use_pretrained = bool(int(words[1]))
                elif words[0] == 'use_predefined_idx':
                    self.use_predefined_idx = bool(int(words[1]))
                elif words[0] == 'pretrained_name':
                    self.pretrained_name = words[1]
                elif words[0] == 'num_subject':
                    self.num_subject = int(words[1])
                elif words[0] == 'num_trial':
                    self.num_trial = int(words[1])
                elif words[0] == 'num_channel':
                    self.num_channel = int(words[1])
                elif words[0] == 'num_length':
                    self.num_length = int(words[1])
                elif words[0] == 'num_class':
                    self.num_class = int(words[1])
                elif words[0] == 'dataset_type':
                    self.dataset_type = words[1]
                elif words[0] == 'num_split':
                    self.num_split = words[1]
                elif words[0] == 'target_subject':
                    self.target_subject = words[1].split(' ')
                    self.target_subject = list(map(int, self.target_subject))
                elif words[0] == 'target_trial':
                    self.target_trial = words[1].split(' ')
                    self.target_trial = list(map(int, self.target_trial))
                elif words[0] == 'train_ratio':
                    self.train_ratio = float(words[1])
                elif words[0] == 'exception_subject':
                    self.exception_subject = words[1].split(' ')
                    self.exception_subject = list(map(int, self.exception_subject))
                elif words[0] == 'exception_trial':
                    self.exception_trial = words[1].split(' ')
                    self.exception_trial = list(map(int, self.exception_trial))
                elif words[0] == 'dataset':
                    self.dataset = words[1]
                elif words[0] == 'model':
                    self.model = words[1]
                elif words[0] == 'target_label':
                    self.target_label = words[1]
                elif words[0] == 'learning_rate':
                    self.learning_rate = float(words[1])
                elif words[0] == 'num_epoch':
                    self.num_epoch = int(words[1])
                elif words[0] == 'batch_size':
                    self.batch_size = int(words[1])
                elif words[0] == 'attack_type':
                    self.attack_type = words[1]
                elif words[0] == 'norm_type':
                    self.norm_type = words[1]
                elif words[0] == 'epsilon':
                    self.epsilon = float(words[1])
                elif words[0] == 'attack_target':
                    self.attack_target = int(words[1])


        f.close()
        self.uap_path = self.uap_path + f'{self.epsilon}' + '/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
        if not os.path.exists(self.uap_path):
            os.makedirs(self.uap_path)

    def PrintConfig(self):
        print('data_path = %s'% self.data_path)
        print('result_path = %s'% self.result_path)
        print('weight_path = %s' % self.weight_path)
        print('uap_path = %s'% self.uap_path)
        print('weight_prefix = %s'% self.weight_prefix)
        print('use_pretrained = %d' % self.use_pretrained)
        print('use_predefined_idx = %d' % self.use_predefined_idx)
        if self.use_pretrained:
            print('weight_prefix = %s' % self.pretrained_name)
        print('split_path = %s' % self.split_path)
        print('num_subject = %d'% self.num_subject)
        print('num_trial = %d'% self.num_trial)
        print('num_channel = %d'% self.num_channel)
        print('num_length = %d' % self.num_length)
        print('num_class = %d' % self.num_class)
        print('dataset type = %s' % self.dataset_type)

        if 'kfold' in self.dataset_type:
            if (len(self.target_subject) > 0):
                print('target_subject = ', end=" ")
                if (self.target_subject[0] == 0):
                    print('all')
                else:
                    print(self.target_subject)
            if (len(self.target_trial) > 0):
                print('target_trial = ', end=" ")
                if (self.target_trial[0] == 0):
                    print('all')
                else:
                    print(self.target_trial)
            print('train_ratio = %f' % self.train_ratio)
        if 'loo' in self.dataset_type:
            if (len(self.exception_subject) > 0):
                print('exception_subject = ', end=" ")
                if (self.exception_subject[0] == 0):
                    print('all')
                else:
                    print(self.exception_subject)
            if (len(self.exception_trial) > 0):
                print('exception_trial = ', end=" ")
                if (self.exception_trial[0] == 0):
                    print('all')
                else:
                    print(self.exception_trial)

        print('dataset = %s' % self.dataset)
        print('model = %s' % self.model)
        print('target_label = %s' % self.target_label)
        print('learning_rate = %f' % self.learning_rate)
        print('num_epoch = %d'% self.num_epoch)
        print('batch_size = %d'% self.batch_size)
        print('attack_type = %s' % self.attack_type)
        if self.attack_type == 'targeted':
            print('attack_target = %d' % self.attack_target)
        print('norm_type = %s' % self.norm_type)
        print('epsilon = %f' % self.epsilon)
        return