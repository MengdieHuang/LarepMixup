"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

from logging import error
from torch import LongTensor
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
import torch
from evaluations.accuracy import EvaluateAccuracy
from utils.saveplt import SaveAccuracyCurve
from utils.saveplt import SaveLossCurve
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch.cuda
import os
import math
import random
import copy
from attacks.advattack import AdvAttack
from clamodels import comparemodels

from tensorboardX import SummaryWriter
from genmodels.mixgenerate import MixGenerate



#-----------------------

def mixup_data(args, exp_result_dir, stylegan2ada_config_kwargs, inputs, targets):
    if args.gen_network_pkl != None:     

        generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
        generate_model.cle_w_train = inputs
        generate_model.cle_y_train = targets

        # print("generate_model.cle_w_train.shape:",generate_model.cle_w_train.shape)
        # print("generate_model.cle_y_train.shape:",generate_model.cle_y_train.shape)

        """
            generate_model.cle_w_train.shape: torch.Size([256, 8, 512])
            generate_model.cle_y_train.shape: torch.Size([256, 8])
        """

        # generate_model.interpolate()

        # print("C inputs:",inputs)
        # print("C targets:",targets)


        mix_w_train, mix_y_train = generate_model.interpolate()                                                                           #   numpy

        # print("G mix_w_train:",mix_w_train)         # CPU tensor
        # print("G mix_y_train:",mix_y_train)        

        generate_model.mix_w_train = mix_w_train    # CPU tensor
        generate_model.mix_y_train = mix_y_train

        mix_x_train, mix_y_train = generate_model.generate()

        # # raise error
        # print("mix_x_train:",mix_x_train)
        # print("mix_y_train:",mix_y_train)

        # print("H mix_x_train:",mix_x_train)         #   CPU tensor
        # print("H mix_y_train:",mix_y_train)

    else:
        raise Exception("There is no gen_network_pkl, please load generative model first!")   
    
    return mix_x_train, mix_y_train

def input_mixup_data(args, raw_img_batch, raw_lab_batch):
    
    # print("raw_img_batch.shape:",raw_img_batch.shape)
    # print("raw_lab_batch.shape:",raw_lab_batch.shape)

    lam = np.random.beta(args.beta_alpha, args.beta_alpha)

    batch_size = raw_img_batch.size()[0]
    index = torch.randperm(batch_size).cuda()   #表示从batch中随机选一个待混合的样本x[index, :]

    mixed_img_batch = lam * raw_img_batch + (1 - lam) * raw_img_batch[index, :]
    mixed_lab_batch = lam * raw_lab_batch + (1 - lam) * raw_lab_batch[index, :]

    # print("mixed_img_batch.shape:",mixed_img_batch.shape)
    # print("mixed_lab_batch.shape:",mixed_lab_batch.shape)
    # """
    # mixed_img_batch.shape: torch.Size([256, 3, 32, 32])
    # mixed_lab_batch.shape: torch.Size([256, 10])
    # """
    return mixed_img_batch,mixed_lab_batch 

def smooth_step(a,b,c,d,epoch_index):
    if epoch_index <= a:        #   <=10
        return 0.01
    if a < epoch_index <= b:    #   10~25
        # return (((epoch_index-a)/(b-a))*(level_m-level_s)+level_s)
        return 0.001
    if b < epoch_index<=c:      #   25~30
        return 0.1
    if c < epoch_index<=d:      #   30~40
        return 0.01
    if d < epoch_index:         #   40~50
        return 0.0001

#--------------
class CustomAlexnet(torch.nn.Module):
    def __init__(self, name='alexnet',n_channels=3, n_outputs=10):
        super(CustomAlexnet, self).__init__()

        self.name = name
        self.num_classes = n_outputs

        self.conv1 = torch.nn.Conv2d(n_channels, 48, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = torch.nn.ReLU()        
        self.lrn = torch.nn.LocalResponseNorm(2)        
        self.pad = torch.nn.MaxPool2d(3, stride=2)
        
        self.batch_norm1 = torch.nn.BatchNorm2d(48, eps=0.001)
        
        self.conv2 = torch.nn.Conv2d(48, 128, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)  
        
        self.batch_norm2 = torch.nn.BatchNorm2d(128, eps=0.001)
        
        self.conv3 = torch.nn.Conv2d(128, 192, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)  
        
        self.batch_norm3 = torch.nn.BatchNorm2d(192, eps=0.001)
        
        self.conv4 = torch.nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)  
        
        self.batch_norm4 = torch.nn.BatchNorm2d(192, eps=0.001)
        
        self.conv5 = torch.nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)  
        
        self.batch_norm5 = torch.nn.BatchNorm2d(128, eps=0.001)
        
        self.fc1 = torch.nn.Linear(1152,512)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0) 
        
        self.drop = torch.nn.Dropout(p=0.5)
        
        self.batch_norm6 = torch.nn.BatchNorm1d(512, eps=0.001)
        
        self.fc2 = torch.nn.Linear(512,256)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0) 
        
        self.batch_norm7 = torch.nn.BatchNorm1d(256, eps=0.001)
        
        self.fc3 = torch.nn.Linear(256,self.num_classes)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0) 
        
        # self.soft = torch.nn.Softmax()    #去掉softmax层
        
    def forward(self, x):
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 128*3*3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)
        #softmax_val = self.soft(logits)

        return logits

class CustomVGG19(torch.nn.Module):
    def __init__(self, name="VGG19", n_channels = 1, n_outputs = 10):
        super(CustomVGG19, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = torch.nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ torch.nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    torch.nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    torch.nn.BatchNorm2d(x),
                    torch.nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [torch.nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return torch.nn.Sequential(*layers)
#--------------

class CustomNet(torch.nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = torch.nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = torch.nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv_1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv_2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = torch.nn.functional.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

class MaggieClassifier:
    r"""    
        class of the target classifier
        attributes:
        self._args
        self._model
        self._loss
        self._optimizer
        self._train_dataloader
        self._test_dataloader
        self._trainset_len
        self._trainbatch_num
        self._exp_result_dir

        methods:
        self.__init__()
        self.__getmodel__()
        self.__gettorchvisionmodel__()
        self.__getlocalmodel__()
        self.__getloss__()
        self.__getoptimizer__()
        self.train()
        self.__trainloop__()
        self.__adjustlearningrate__()
    """

    def __init__(self, args, learned_model=None) -> None:                 # 双下划线表示只有Classifier类本身可以访问   ->后是对函数返回值的注释，None表明无返回值
        print('initlize classifier')

        # initilize the parameters
        self._args = args

        # initilize the model architecture
        if learned_model == None:
            print("learned calssify model = None")
            self._model = self.__getmodel__()
        else:
            print("learned calssify model != None")
            self._model = learned_model             #   浅拷贝
        
        # initilize the loss function
        self._lossfunc = self.__getlossfunc__()
        
        # initilize the optimizer
        self._optimizer = self.__getoptimizer__()
    
    def model(self) -> "torchvision.models or CustomNet":
        return self._model

    def __getmodel__(self) -> "torchvision.models or CustomNet":
        model_name = self._args.cla_model

        # torchvisionmodel_dict = ['resnet34','resnet50','vgg19','densenet169','alexnet','inception_v3']    # 少 alexnet
        # torchvisionmodel_dict = ['resnet34','resnet50','vgg19','densenet169','inception_v3','resnet18','googlenet']
        torchvisionmodel_dict = ['resnet34','resnet50','densenet169','inception_v3','resnet18','googlenet'] # 少 vgg19
        comparemodel_dict = ['preactresnet18','preactresnet34','preactresnet50']    #   ,'wideresnet2810'
        if model_name in torchvisionmodel_dict:
            model = self.__gettorchvisionmodel__()      #   加载torchvision库model
        elif model_name in comparemodel_dict:
            model = self.__getcomparemodel__()
        else:   # alexnet, vgg19
            if self._args.img_size <= 32:           #   32的数据用自定义的alexnet训练
                model = self.__getlocalmodel__()
            elif self._args.img_size > 32:
                model = self.__gettorchvisionmodel__()      #   加载torchvision库model
        return model

    def __gettorchvisionmodel__(self) ->"torchvision.models":
        print('使用pytorch库模型')
        model_name = self._args.cla_model
        classes_number = self._args.n_classes
        pretrain_flag = self._args.pretrained_on_imagenet
        img_channels = self._args.channels
        print("model_name:",model_name)
        print("classes_number:",classes_number)
        print("pretrain_flag:",pretrain_flag)   #  pretrain_flag: False
        print("img_channels:",img_channels)   #  pretrain_flag: False

        if pretrain_flag is True and self._args.dataset == 'imagenet':
            #   加载torchvision库中的原始模型
            torchvisionmodel =  torchvision.models.__dict__[model_name](pretrained=pretrain_flag)
            last = list(torchvisionmodel.named_modules())[-1][1]
            print('original torchvisionmodel.last:',last)        #   torchvisionmodel.last: Linear(in_features=4096, out_features=10, bias=True)

        else:
            # #   加载torchvision库中的原始模型
            # torchvisionmodel =  torchvision.models.__dict__[model_name](pretrained=pretrain_flag)
                
            # #   获取原始模型的最后一层信息
            # last_name = list(torchvisionmodel._modules.keys())[-1]
            # last_module = torchvisionmodel._modules[last_name]
            # # print('last_name:',last_name)               #   alexnet last_name: classifier
            # # print('last_module:',last_module)           #   alexnet last_module: Sequential
            
            # #   修改最后一层信息
            # if isinstance(last_module, torch.nn.Linear):                                #   resnet,inception,googlenet、shfflenetv2 最后一层是nn.linear
            #     n_features = last_module.in_features
            #     torchvisionmodel._modules[last_name] = torch.nn.Linear(n_features, classes_number)

            # elif isinstance(last_module, torch.nn.Sequential):                          #   alexnet、vgg、mobilenet、mnasnet 最后一模块是nn.Sequential
            #     # 获取最后一模块的最后一层信息
            #     # seq_last_name = list(torchvisionmodel._modules.keys())[-1]
            #     # seq_last_module = torchvisionmodel._modules[seq_last_name]
            #     seq_last_name = list(last_module._modules.keys())[-1]
            #     seq_last_module = last_module._modules[seq_last_name]            
            #     # print('seq_last_name:',seq_last_name)                       #   alexnet seq_last_name: 6
            #     # print('seq_last_module:',seq_last_module)                   #   seq_last_module: Linear(in_features=4096, out_features=1000, bias=True)

            #     n_features = seq_last_module.in_features
            #     last_module._modules[seq_last_name] = torch.nn.Linear(n_features, classes_number)
            
            torchvisionmodel =  torchvision.models.__dict__[model_name](pretrained=pretrain_flag, num_classes = classes_number, n_channels = img_channels)

            last = list(torchvisionmodel.named_modules())[-1][1]
            print('modified torchvisionmodel.last:',last)        #   torchvisionmodel.last: Linear(in_features=4096, out_features=10, bias=True)
            # raise error 

        # raise error
        return torchvisionmodel

    def __getcomparemodel__(self):
        print('使用文献对比模型')
        model_name = self._args.cla_model
        classes_number = self._args.n_classes
        pretrain_flag = self._args.pretrained_on_imagenet
        img_channels = self._args.channels
        print("model_name:",model_name)
        print("classes_number:",classes_number)
        print("pretrain_flag:",pretrain_flag)   #  pretrain_flag: False
        print("img_channels:",img_channels)   #  pretrain_flag: False

        net = comparemodels.__dict__[model_name]()

        #   获取原始模型的最后一层信息
        last_name = list(net._modules.keys())[-1]
        last_module = net._modules[last_name]
        print('last_name:',last_name)               #   last_name: fc3
        print('last_module:',last_module)           #   last_module: Linear(in_features=256, out_features=10, bias=True)
        # raise error            
        return net

    def __getlocalmodel__(self)->"CustomNet":
        print('使用自定义模型')
        # 3个输入变量 模型name, 类别num, 预训练flag
        model_name = self._args.cla_model
        classes_number = self._args.n_classes
        pretrain_flag = self._args.pretrained_on_imagenet
        data_channels = self._args.channels
        print("model_name:",model_name)                         #   model_name: alexnet
        print("classes_number:",classes_number)
        print("pretrain_flag:",pretrain_flag)                   #  pretrain_flag: False
        print("self._args.channels:",self._args.channels)       #   self._args.channels: 3

        if model_name == 'alexnet':
            local_model = CustomAlexnet(name='alexnet',n_channels=data_channels, n_outputs=classes_number)
        elif model_name == 'vgg19':
            local_model = CustomVGG19(name='VGG19',n_channels=data_channels, n_outputs=classes_number)
        else:
            local_model = CustomNet()

        #   获取原始模型的最后一层信息
        last_name = list(local_model._modules.keys())[-1]
        last_module = local_model._modules[last_name]
        print('last_name:',last_name)               #   last_name: fc3
        print('last_module:',last_module)           #   last_module: Linear(in_features=256, out_features=10, bias=True)
        # raise error            
        return local_model

    def __getlossfunc__(self):
        # torch.nn.L1Loss
        # torch.nn.KLDivLoss
        # torch.nn.SmoothL1Loss
        # torch.nn.SoftMarginLoss
        # torch.nn.LocalResponseNorm
        # torch.nn.MultiMarginLoss
        # torch.nn.CrossEntropyLoss
        # torch.nn.BCEWithLogitsLoss
        # torch.nn.MarginRankingLoss
        # torch.nn.TripletMarginLoss
        # torch.nn.HingeEmbeddingLoss
        # torch.nn.CosineEmbeddingLoss
        # torch.nn.MultiLabelMarginLoss
        # torch.nn.MultiLabelSoftMarginLoss
        # torch.nn.AdaptiveLogSoftmaxWithLoss
        # torch.nn.TripletMarginWithDistanceLoss
        lossfunc = torch.nn.CrossEntropyLoss()
        return lossfunc
    
    def __getoptimizer__(self):
        # torch.optim.Adadelta()
        # torch.optim.Adagrad()
        # torch.optim.Adam()
        # torch.optim.Adamax()
        # torch.optim.AdamW()
        # torch.optim.ASGD()
        # torch.optim.LBFGS()
        # torch.optim.RMSprop()
        # torch.optim.Rprop()
        # torch.optim.SGD()
        # torch.optim.SparseAdam()
        # torch.optim.Optimizer()
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._args.lr)

        # optimizer = torch.optim.SGD(params=self._model.parameters(), lr=self._args.lr, momentum=0.9,  weight_decay=1e-4)
        return optimizer

    def train(self,train_dataloader,test_dataloader,exp_result_dir, train_mode) -> "torchvision.models or CustomNet":

        # initilize the dataloader
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader 
        self._trainset_len = len(self._train_dataloader.dataset)
        self._trainbatch_num = len(self._train_dataloader)
        self._exp_result_dir = exp_result_dir
        print("self._trainset_len:",self._trainset_len)                 #   77237
        print("self._trainbatch_num:",self._trainbatch_num)
        print("self._testset_len:",len( self._test_dataloader.dataset))     #   self._testset_len: 3000
        # print("self._train_dataloader.dataset[0][0][0].shape:",self._train_dataloader.dataset[0][0][0].shape)
        # print("self._test_dataloader.dataset[0][0][0].shape:",self._test_dataloader.dataset[0][0][0].shape)

        # raise error

        self._exp_result_dir = os.path.join(self._exp_result_dir,f'train-{self._args.dataset}-dataset')
        os.makedirs(self._exp_result_dir,exist_ok=True)    

        if torch.cuda.is_available():
            self._lossfunc.cuda()
            self._model.cuda()
        
        global_train_acc, global_test_acc, global_train_loss, global_test_loss = self.__trainloop__()
        
        if train_mode == "std-train":
            torch.save(self._model,f'{self._exp_result_dir}/standard-trained-classifier-{self._args.cla_model}-on-clean-{self._args.dataset}-finished.pkl')
            accuracy_png_name = f'standard trained classifier {self._args.cla_model} accuracy on clean {self._args.dataset}'
            loss_png_name = f'standard trained classifier {self._args.cla_model} loss on clean {self._args.dataset}'
        
        elif train_mode == "adv-train":     
            torch.save(self._model,f'{self._exp_result_dir}/adversarial-trained-classifier-{self._args.cla_model}-on-adv-{self._args.dataset}-finished.pkl')
            accuracy_png_name = f'adversarial trained classifier {self._args.cla_model} accuracy on adversarial {self._args.dataset}'
            loss_png_name = f'adversarial trained classifier {self._args.cla_model} loss on adversarial {self._args.dataset}'

        SaveAccuracyCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_acc, global_test_acc, accuracy_png_name)

        SaveLossCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_loss, global_test_loss, loss_png_name)

        return self._model

    def __trainloop__(self):

        global_train_acc = []
        global_test_acc = []
        global_train_loss = []
        global_test_loss = []

        for epoch_index in range(self._args.epochs):
            
            self.__adjustlearningrate__(epoch_index)     

            epoch_correct_num = 0
            epoch_total_loss = 0

            for batch_index, (images, labels) in enumerate(self._train_dataloader):

                batch_imgs = images.cuda()
                batch_labs = labels.cuda()
                self._optimizer.zero_grad()

                self._model.train()     #   train mode
                if self._args.cla_model == 'inception_v3':
                    output, aux = self._model(batch_imgs)
                elif self._args.cla_model == 'googlenet':
                    output, aux1, aux2 = self._model(batch_imgs)
                else:
                    output = self._model(batch_imgs)

                # print("output:",output)                                     #   output: tensor([[-0.2694,  0.1577,  0.4321,  ...,  0.0562,  0.3836,  0.8319],
                # print("output.shape:",output.shape)                         #   output.shape: torch.Size([256, 10])
                # softmax_output = torch.nn.functional.softmax(output, dim = 1)
                # print("softmax_output:",softmax_output)                     #   softmax_output: tensor([[0.0128, 0.1096, 0.0726,  ..., 0.1264, 0.0353, 0.3227],
                # print("softmax_output.shape:",softmax_output.shape)         #   softmax_output.shape: torch.Size([256, 10])              
                # # raise error

                batch_loss = self._lossfunc(output,batch_labs)
                # batch_loss = self._lossfunc(softmax_output,batch_labs)
                # raise error
                batch_loss.backward()
                self._optimizer.step()

                _, predicted_label_index = torch.max(output.data, 1)   
                # _, predicted_label_index = torch.max(softmax_output.data, 1)    

                batch_correct_num = (predicted_label_index == batch_labs).sum().item()     
                epoch_correct_num += batch_correct_num                                     
                epoch_total_loss += batch_loss

                print("[Epoch %d/%d] [Batch %d/%d] [Batch classify loss: %f] " % (epoch_index+1, self._args.epochs, batch_index+1, len(self._train_dataloader), batch_loss.item()))

            #--------当前epoch分类模型在当前训练集epoch上的准确率和loss-------------            
            epoch_train_accuarcy = epoch_correct_num / len(self._train_dataloader.dataset)      #    除以epoch里的样本总数
            epoch_train_loss = epoch_total_loss / len(self._train_dataloader)                   #   除以样本总数

            global_train_loss.append(epoch_train_loss)
            global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list                  

            #--------当前epoch分类模型在整体测试集上的准确率------------- 
            epoch_test_accuracy, epoch_test_loss = EvaluateAccuracy(self._model, self._lossfunc, self._test_dataloader,self._args.cla_model)
            global_test_acc.append(epoch_test_accuracy)   
            global_test_loss.append(epoch_test_loss)

            # print(f'{epoch_index:04d} epoch classifier accuary on the current epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
            # print(f'{epoch_index:04d} epoch classifier loss on the current epoch training examples:{epoch_train_loss:.4f}' )   
            print(f'{epoch_index+1:04d} epoch classifier accuary on the entire testing examples:{epoch_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index+1:04d} epoch classifier loss on the entire testing examples:{epoch_test_loss:.4f}' )  
            
            # if (epoch_index+1) % 11== 0 and epoch_index > 0:
            if (epoch_index+1)  >= 9:
                torch.save(self._model,f'{self._exp_result_dir}/standard-trained-classifier-{self._args.cla_model}-on-clean-{self._args.dataset}-epoch-{epoch_index+1:04d}.pkl')
            
            #-------------tensorboard实时画图-------------------
            tensorboard_log_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc')
            os.makedirs(tensorboard_log_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_acc = SummaryWriter(log_dir = tensorboard_log_acc_dir, comment= '-'+'testacc')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_acc.add_scalar(tag = "epoch_test_acc", scalar_value = epoch_test_accuracy, global_step = epoch_index + 1 )
            writer_acc.close()
            # raise error
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss')
            os.makedirs(tensorboard_log_loss_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_loss = SummaryWriter(log_dir = tensorboard_log_loss_dir, comment= '-'+'testloss')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_loss.add_scalar(tag = "epoch_test_loss", scalar_value = epoch_test_loss, global_step = epoch_index + 1 )
            writer_loss.close()
            # raise error
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_train_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-train-acc')
            os.makedirs(tensorboard_log_train_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_train_acc = SummaryWriter(log_dir = tensorboard_log_train_acc_dir, comment= '-'+'trainacc')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_train_acc.add_scalar(tag = "epoch_train_acc", scalar_value = epoch_train_accuarcy, global_step = epoch_index + 1 )
            writer_train_acc.close()
            # raise error
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_train_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-train-loss')
            os.makedirs(tensorboard_log_train_loss_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_train_loss = SummaryWriter(log_dir = tensorboard_log_train_loss_dir, comment= '-'+'trainloss')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_train_loss.add_scalar(tag = "epoch_train_loss", scalar_value = epoch_train_loss, global_step = epoch_index + 1 )
            writer_train_loss.close()
            # raise error
            #--------------------------------------------------


        return global_train_acc, global_test_acc, global_train_loss, global_test_loss
    
    def evaluatefromdataloader(self,model,test_dataloader) -> None:
        if torch.cuda.is_available():
            self._lossfunc.cuda()
            # self._model.cuda()    #             check
            model.cuda()
        test_accuracy, test_loss = EvaluateAccuracy(model, self._lossfunc, test_dataloader,self._args.cla_model)     
        # print(f'classifier *accuary* on testset:{test_accuracy * 100:.4f}%' ) 
        # print(f'classifier *loss* on testset:{test_loss}' ) 
        #  
        return test_accuracy, test_loss

    def artmodel(self)->"PyTorchClassifier":
        # initilize the art format attack model
        self._artmodel = self.__getartmodel__()
        return self._artmodel

    def __getartmodel__(self) -> "PyTorchClassifier":
        if torch.cuda.is_available():
            self._lossfunc.cuda()
            self._model.cuda()      
        
        data_raw = False                                        #   是否在之前对数据集进行过归一化
        if data_raw == True:
            min_pixel_value = 0.0
            max_pixel_value = 255.0
        else:
            min_pixel_value = 0.0
            max_pixel_value = 1.0        

        artmodel = PyTorchClassifier(
            model=self._model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=self._lossfunc,
            optimizer=self._optimizer,
            input_shape=(self._args.channels, self._args.img_size, self._args.img_size),
            nb_classes=self._args.n_classes,
        )             
        return artmodel

    def evaluatefromtensor(self, classifier, x_set:Tensor, y_set:Tensor):
        classifier.eval()   #   eval mode
        if torch.cuda.is_available():
            classifier.cuda()             
        
        batch_size = self._args.batch_size
        testset_total_num = len(x_set)
        batch_num = int( np.ceil( int(testset_total_num) / float(batch_size) ) )
        cla_model_name=self._args.cla_model

        # print("x_set.shape:",x_set.shape)           #   x_set.shape: torch.Size([26032, 3, 32, 32])
        # print("y_set.shape:",y_set.shape)           #   y_set.shape: torch.Size([26032])
        # print("testset_total_num:",testset_total_num)       #   testset_total_num: 26032
        # print("batch_num:",batch_num)                   #   batch_num: 
        # print("batch_size:",batch_size)       #  
        # print("cla_model_name:",cla_model_name)       #   cla_model_name: 

        eva_loss = torch.nn.CrossEntropyLoss()
        epoch_correct_num = 0
        epoch_total_loss = 0

        # batch_same_num = 0

        for batch_index in range(batch_num):                                                #   进入batch迭代 共有num_batch个batch
            right_index = min((batch_index + 1) * batch_size, testset_total_num)
            # print("right_index:",right_index)
            images = x_set[batch_index * batch_size : right_index]
            labels = y_set[batch_index * batch_size : right_index]                                                

            imgs = images.cuda()
            labs = labels.cuda()
            eva_loss = eva_loss.cuda()

            
            with torch.no_grad():

                if cla_model_name == 'inception_v3':
                    output, aux = classifier(imgs)
                
                elif cla_model_name == 'googlenet':
                    if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'svhn' or self._args.dataset == 'kmnist' or self._args.dataset == 'cifar10':  #   只有imagenet和svhn kmnist搭配googlenet时是返回一个值
                        output = classifier(imgs)
                    else:
                        output, aux1, aux2 = classifier(imgs)
                else:   # resnet alexent densenet 
                    output = classifier(imgs)         
                                
                loss = eva_loss(output,labs)
                _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
                
                assert len(predicted_label_index) == len(labs)
                # batch_same_num = (predicted_label_index == labs).sum().item()
                batch_same_num = predicted_label_index.cpu().eq(labs.cpu()).sum()

                # same_lables += student_labels.cpu().eq(labels.cpu()).sum()
                epoch_correct_num += batch_same_num
                epoch_total_loss += loss

        # print("epoch_correct_num:",epoch_correct_num)

        test_accuracy = epoch_correct_num / testset_total_num
        test_loss = epoch_total_loss / batch_num                  
        classifier.train()
        return test_accuracy, test_loss

    #   get clean set tensor format
    def getrawset(self,dataloader)->"Tensor":
        # self._train_dataloader = train_dataloader
        xset_tensor, yset_tensor = self.__getrawsettensor__(dataloader)
        return xset_tensor, yset_tensor
    
    def __getrawsettensor__(self,dataloader)->"Tensor":

        xset_tensor  = self.__getxsettensor__(dataloader)
        yset_tensor = self.__getysettensor__(dataloader)

        return xset_tensor, yset_tensor
    
    def __getxsettensor__(self,dataloader)->"Tensor":

        # print("dataloader.dataset.data[0]:",dataloader.dataset.data[0])             #   dataloader.dataset.data[0]: [[[ 59  62  63]

        if self._args.dataset == 'cifar10':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                         

        elif self._args.dataset == 'cifar100':
            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
                                        

        elif self._args.dataset == 'imagenet':
            jieduan_num = 1000

            xset_tensor = []
            # for img_index in range(len(dataloader.dataset)):
            for img_index in range(jieduan_num):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
            
        elif self._args.dataset == 'svhn':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)   

        elif self._args.dataset == 'kmnist':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)  

        # return xset_tensor.cuda()                                       #   xset_tensor原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接
        return xset_tensor.cpu()

    def __getysettensor__(self,dataloader)->"Tensor":

        if self._args.dataset == 'cifar10':
        #     y_ndarray = dataloader.dataset.targets
        #     print("y_ndarray.type:", type(y_ndarray))

            # y_ndarray = y_ndarray[:jieduan_num]

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           #   list型转为tensor
            # print("yset_tensor.type:", type(yset_tensor))                   #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)

        elif self._args.dataset == 'cifar100':

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           
            # print("yset_tensor.type:", type(yset_tensor))                                         
            # print("yset_tensor.shape:", yset_tensor.shape)

        elif self._args.dataset == 'imagenet':
            jieduan_num = 1000
            # y_ndarray = []       
            # datasetset_len = len(dataloader.dataset)
            # print('datasetset len:',datasetset_len)

            # for index in range(jieduan_num):
            # # for index in range(datasetset_len):

            #     _, label = dataloader.dataset.__getitem__(index)
            #     y_ndarray.append(label)      

            yset_tensor = []
            # for img_index in range(len(dataloader.dataset)):
            for img_index in range(jieduan_num):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           
            # print("yset_tensor.type:", type(yset_tensor))                       #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)                      #   yset_tensor.shape: torch.Size([1000])

        elif self._args.dataset == 'svhn':
        #     y_ndarray = dataloader.dataset.targets
        #     print("y_ndarray.type:", type(y_ndarray))

            # y_ndarray = y_ndarray[:jieduan_num]

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           #   list型转为tensor
            # print("yset_tensor.type:", type(yset_tensor))                   #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)

        elif self._args.dataset == 'kmnist':
        #     y_ndarray = dataloader.dataset.targets
        #     print("y_ndarray.type:", type(y_ndarray))

            # y_ndarray = y_ndarray[:jieduan_num]

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           #   list型转为tensor
            # print("yset_tensor.type:", type(yset_tensor))                   #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)


        # return yset_tensor.cuda()       #   yset_tensor 原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接
        return yset_tensor.cpu()       #   yset_tensor 原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接

    #   get adv set tensor format
    def getadvset(self,adv_dataset_path):
        adv_xset_tensor, adv_yset_tensor = self.__getadvsettensor__(adv_dataset_path)
        return adv_xset_tensor, adv_yset_tensor     
        
    def __getadvsettensor__(self,adv_dataset_path):

        if self._args.perceptualattack == False:
            file_dir=os.listdir(adv_dataset_path)
            file_dir.sort()
            # '00000000-adv-3-cat.npz'
            filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[9:12] == 'adv']           

            adv_xset_tensor = []
            adv_yset_tensor = []

            for index, filename in enumerate(filenames):
                adv_npz_path = os.path.join(adv_dataset_path,filename)

                load_adv_img = np.load(adv_npz_path)['w']            
                load_adv_img = torch.tensor(load_adv_img)
                
                load_adv_label = int(filename[13:14])               #   从文件名中读出对抗样本的label信息
                load_adv_label = torch.tensor(load_adv_label)

                adv_xset_tensor.append(load_adv_img)
                adv_yset_tensor.append(load_adv_label)

            adv_xset_tensor = torch.stack(adv_xset_tensor)                                                                         
            adv_yset_tensor = torch.stack(adv_yset_tensor)   

            # return adv_xset_tensor.cuda(), adv_yset_tensor.cuda()     
            return adv_xset_tensor.cpu(), adv_yset_tensor.cpu()  

        elif self._args.perceptualattack == True:
            file_dir=os.listdir(adv_dataset_path)
            file_dir.sort()
            # '00000000-adv-3-cat.npz'
            filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[9:12] == 'per']           

            adv_xset_tensor = []
            adv_yset_tensor = []

            for index, filename in enumerate(filenames):
                adv_npz_path = os.path.join(adv_dataset_path,filename)

                load_adv_img = np.load(adv_npz_path)['w']            
                load_adv_img = torch.tensor(load_adv_img)
                
                load_adv_label = int(filename[13:14])               #   从文件名中读出对抗样本的label信息
                load_adv_label = torch.tensor(load_adv_label)

                adv_xset_tensor.append(load_adv_img)
                adv_yset_tensor.append(load_adv_label)

            adv_xset_tensor = torch.stack(adv_xset_tensor)                                                                         
            adv_yset_tensor = torch.stack(adv_yset_tensor)   

            # return adv_xset_tensor.cuda(), adv_yset_tensor.cuda()     
            return adv_xset_tensor.cpu(), adv_yset_tensor.cpu()    

    #   get representation set tensor format
    def getproset(self, pro_dataset_path):
        pro_wset_tensor, pro_yset_tensor = self.__getprosettensor__(pro_dataset_path)
        return pro_wset_tensor, pro_yset_tensor     
        
    def __getprosettensor__(self,pro_dataset_path):
        file_dir=os.listdir(pro_dataset_path)
        file_dir.sort()
        # '00000000-6-frog-projected_w.npz'
        # '00000000-6-frog-label.npz'

        npzfile_name = []
        for name in file_dir:                                                                                                   #   选择指定目录下的.npz文件名
            if os.path.splitext(name)[-1] == '.npz':
                npzfile_name.append(name)                                                                                       #   name指代了list中的object 
                # 00000000-1-1-projected_w.npz
                # 00000000-1-1-label.npz

        projected_w_npz_paths =[]
        label_npz_paths = []
        for name in npzfile_name:
            if name[-15:-4] == 'projected_w':   #   倒数几位
                projected_w_npz_paths.append(f'{self._args.projected_dataset}/{name}')

            elif name[-9:-4] == 'label':
                label_npz_paths.append(f'{self._args.projected_dataset}/{name}')


        pro_wset_tensor = []
        pro_yset_tensor = []
        device = torch.device('cuda')

        for projected_w_path in projected_w_npz_paths:                                                                                   
            w = np.load(projected_w_path)['w']
            w = torch.tensor(w, device=device)                                                                                 
            w = w[-1]                                                                                       #   w.shape: torch.Size([1, 8,512]))         
            pro_wset_tensor.append(w)        

        pro_wset_tensor = torch.stack(pro_wset_tensor)           
        # print("pro_wset_tensor.shape:",pro_wset_tensor.shape)                                         #   projected_w_set_x.shape: torch.Size([37, 8, 512])
                                                                                                            #   stl10 projected_w_set_x.shape: torch.Size([38, 10, 512])

        for label_npz_path in label_npz_paths:                                                                                  
            y = np.load(label_npz_path)['w']
            y = torch.tensor(y, device=device)                                                                                 
            y = y[-1]                                                                                       #   y.shape: torch.Size([1, 8]))
            pro_yset_tensor.append(y)

        pro_yset_tensor = torch.stack(pro_yset_tensor)           
        # print("pro_yset_tensor.shape:",pro_yset_tensor.shape)                                         #   projected_w_set_y.shape: torch.Size([37, 8])  
        # print("pro_yset_tensor[0]:",pro_yset_tensor[0])                                         #   projected_w_set_y.shape: torch.Size([37, 8])  

        # pro_yset_tensor = torch.nn.functional.one_hot(pro_yset_tensor, self._args.n_classes).float().to(device)                           
        # print("pro_yset_tensor.shape:",pro_yset_tensor.shape)                                         #   projected_w_set_y.shape: torch.Size([37, 8, 10])
        # print("pro_yset_tensor[0]:",pro_yset_tensor[0])                                         #   projected_w_set_y.shape: torch.Size([37, 8])  

        # return pro_wset_tensor.cuda(), pro_yset_tensor.cuda()     
        return pro_wset_tensor.cpu(), pro_yset_tensor.cpu()

    def adversarialtrain(self,
        args,
        cle_x_train,
        cle_y_train,
        cle_x_test,
        cle_y_test,

        x_train_adv,
        y_train_adv,
        x_test_adv, 
        y_test_adv, 
        classify_model: "PyTorchClassifier",
        exp_result_dir
    ):

        # print("cle_x_train.type:",type(cle_x_train))            #   cle_x_train.type: <class 'torch.Tensor'>
        # print("cle_x_train.shape:",cle_x_train.shape)           #   cle_x_train.shape: torch.Size([50000, 3, 32, 32])
        # print("x_train_adv.type:",type(x_train_adv))            #   x_train_adv.type: <class 'torch.Tensor'>
        # print("x_train_adv.shape:",x_train_adv.shape)           #   x_train_adv.shape: torch.Size([50000, 3, 32, 32])

        # print("cle_y_train.type:",type(cle_y_train))            #   cle_y_train.type: <class 'torch.Tensor'>
        # print("cle_y_train.shape:",cle_y_train.shape)           #   cle_y_train.shape: torch.Size([50000])
        # print("y_train_adv.type:",type(y_train_adv))            #   y_train_adv.type: <class 'torch.Tensor'>
        # print("y_train_adv.shape:",y_train_adv.shape)           #   y_train_adv.shape: torch.Size([50000])


        # #扩增对抗样本: 50000 x_train_adv
        # aug_x_train = cle_x_train
        # aug_y_train = cle_y_train

        #   #扩增对抗样本: 50000 x_train_adv
        # aug_x_train = x_train_adv
        # aug_y_train = y_train_adv
        
        #   扩增对抗样本: 50000 cle_x_train + 50000 x_train_adv
        # print("cle_x_train.dtype:",cle_x_train.dtype)       #   cle_x_train.dtype: torch.float32
        # print("cle_y_train.dtype:",cle_y_train.dtype)       #   cle_y_train.dtype: torch.int64
        # print("x_train_adv.dtype:",x_train_adv.dtype)       #   x_train_adv.dtype: torch.float32
        # print("y_train_adv.dtype:",y_train_adv.dtype)       #   y_train_adv.dtype: torch.int64

        if args.aug_adv_num is None:
            select_num = len(cle_y_train)
        else:
            select_num = args.aug_adv_num

        cle_x_train = cle_x_train[:select_num]
        cle_y_train = cle_y_train[:select_num]
        x_train_adv = x_train_adv[:select_num]
        y_train_adv = y_train_adv[:select_num]

        # cle_x_train.cuda()
        # cle_y_train.cuda()
        # x_train_adv.cuda()
        # y_train_adv.cuda() 

        # print("cle_x_train.shape:",cle_x_train.shape)       #   cle_x_train.shape: torch.Size([73257, 3, 32, 32])
        # print("cle_y_train.shape:",cle_y_train.shape)       #   
        # print("x_train_adv.shape:",x_train_adv.shape)       #   x_train_adv.shape: torch.Size([5000, 3, 32, 32])
        # print("y_train_adv.shape:",y_train_adv.shape)       #   

        aug_x_train = torch.cat([cle_x_train, x_train_adv], dim=0)
        aug_y_train = torch.cat([cle_y_train, y_train_adv], dim=0)
      

        # print("aug_x_train.type:",type(aug_x_train))            #   aug_x_train.type: <class 'torch.Tensor'>
        # print("aug_x_train.shape:",aug_x_train.shape)           #   aug_x_train.shape: torch.Size([78257, 3, 32, 32])
        # print("aug_y_train.type:",type(aug_y_train))            #   aug_y_train.type: <class 'torch.Tensor'>
        # print("aug_y_train.shape:",aug_y_train.shape)           #   aug_y_train.shape: torch.Size([78257])
        # raise Exception("maggie error")    

        # tensor转numpy
        aug_x_train = aug_x_train.cpu().numpy()
        aug_y_train = aug_y_train.cpu().numpy()
        # print("aug_x_train.type:",type(aug_x_train))            #   aug_x_train.type: <class 'numpy.ndarray'>
        # print("aug_x_train.shape:",aug_x_train.shape)           #   aug_x_train.shape: (100000, 3, 32, 32)
        # print("aug_y_train.type:",type(aug_y_train))            #   aug_y_train.type: <class 'numpy.ndarray'>
        # print("aug_y_train.shape:",aug_y_train.shape)           #   aug_y_train.shape: (100000,)
    
        # classify_model.fit(aug_x_train, aug_y_train, x_test_adv, y_test_adv, exp_result_dir, args, nb_epochs=args.epochs, batch_size=args.batch_size)            

        classify_model.fit(aug_x_train, aug_y_train, cle_x_test, cle_y_test, x_test_adv, y_test_adv, exp_result_dir, args, nb_epochs=args.epochs, batch_size=args.batch_size)

    def mmat(self,
        args,
        cle_x_train,
        cle_y_train,
        x_train_mix,
        y_train_mix,

        cle_x_test, 
        cle_y_test,
        x_test_adv, 
        y_test_adv, 

        exp_result_dir,
        classify_model = None #: ART中的"PyTorchClassifier"
    ):
        # print("cle_x_test.shape:",cle_x_test.shape)
        # print("cle_y_test.shape:",cle_y_test.shape)
        cle_y_train_onehot = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float().cuda()  

        if args.aug_num is None or args.aug_mix_rate is None:
            # select_num = len(cle_y_train)
            raise Exception("input aug_num and aug_mix_rate")
        else:
            aug_num = args.aug_num           #   共要扩增多少样本
            aug_rate = args.aug_mix_rate
            select_cle_num = int( (1-aug_rate) * aug_num )
            select_mix_num = int( aug_rate * aug_num )

        if aug_rate == 0:
            print("*only using clean samples*")
            # select_cle_num = int( (1-aug_rate) * aug_num ) 
            aug_x_train = cle_x_train[:select_cle_num]
            aug_y_train = cle_y_train_onehot[:select_cle_num]

        elif aug_rate == 1:
            print("*only using mixed samples*")
            # select_mix_num = int( aug_rate * aug_num )
            aug_x_train = x_train_mix[:select_mix_num]
            aug_y_train = y_train_mix[:select_mix_num]

        elif aug_rate == 0.5:
            print("*using clean sampels and mixed samples*")
            aug_rate = 0.5

            cle_x_train         = cle_x_train[:select_cle_num]
            cle_y_train_onehot  = cle_y_train_onehot[:select_cle_num]
            x_train_mix         = x_train_mix[:select_mix_num]
            y_train_mix         = y_train_mix[:select_mix_num]

            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            print("x_train_mix.shape:",x_train_mix.shape)
            print("cle_y_train_onehot.shape:",cle_y_train_onehot.shape)

            aug_x_train = torch.cat([cle_x_train, x_train_mix], dim=0)
            aug_y_train = torch.cat([cle_y_train_onehot, y_train_mix], dim=0)

        elif args.aug_mix_rate is None:
            raise Exception("input augmentation rate please")
        
        aug_x_train = aug_x_train.cpu().numpy()
        aug_y_train = aug_y_train.cpu().numpy()
        print("aug_x_train.shape:",aug_x_train.shape)           #   aug_x_train.shape: (42588, 3, 32, 32)                                                    
        print("aug_y_train.shape:",aug_y_train.shape) 
        # raise error                                                              
        print(f"use {select_cle_num}/{aug_num} clean sampels，{select_mix_num}/{aug_num} mixed samples")
        
        self.__softtrain__(aug_x_train, aug_y_train, cle_x_test, cle_y_test,  x_test_adv, y_test_adv, exp_result_dir)
        
        # classify_model.fit_softlabel( aug_x_train, aug_y_train, cle_x_test, cle_y_test, x_test_adv, y_test_adv, exp_result_dir, args, nb_epochs=args.epochs,batch_size=args.batch_size)

    def __softtrain__(self, aug_x_train, aug_y_train, cle_x_test, cle_y_test, x_test_adv, y_test_adv,exp_result_dir):
            self._train_tensorset_x = torch.tensor(aug_x_train)
            self._train_tensorset_y = torch.tensor(aug_y_train)

            self._adv_test_tensorset_x = torch.tensor(x_test_adv)
            self._adv_test_tensorset_y = torch.tensor(y_test_adv)

            self._cle_test_tensorset_x = torch.tensor(cle_x_test)
            self._cle_test_tensorset_y = torch.tensor(cle_y_test)



            self._exp_result_dir = exp_result_dir
            if self._args.defense_mode == "mmat":
                self._exp_result_dir = os.path.join(self._exp_result_dir,f'mmat-{self._args.dataset}-dataset')

            elif self._args.defense_mode == "at":
                self._exp_result_dir = os.path.join(self._exp_result_dir,f'at-{self._args.dataset}-dataset')
            os.makedirs(self._exp_result_dir,exist_ok=True)            


            if torch.cuda.is_available():
                self._lossfunc.cuda()
                self._model.cuda()          #   self._model在初始化时被赋值为了读入的模型

            global_train_acc, global_adv_test_acc, global_cle_test_acc, global_train_loss, global_adv_test_loss, global_cle_test_loss = self.__traintensorsetloop__()
            
            # if self._args.defense_mode == "mmat":
            #     accuracy_png_name_adv   = f'mmat trained classifier {self._args.cla_model} accuracy on adversarial {self._args.dataset}'
            #     loss_png_name_adv       = f'mmat trained classifier {self._args.cla_model} loss on adversarial {self._args.dataset}'   
            #     accuracy_png_name_cle   = f'mmat trained classifier {self._args.cla_model} accuracy on clean {self._args.dataset}'
            #     loss_png_name_cle       = f'mmat trained classifier {self._args.cla_model} loss on clean {self._args.dataset}'               

            # elif self._args.defense_mode == "at":
            #     accuracy_png_name_adv   = f'adversarial trained classifier {self._args.cla_model} accuracy on adversarial {self._args.dataset}'
            #     loss_png_name_adv       = f'adversarial trained classifier {self._args.cla_model} loss on adversarial {self._args.dataset}'   
            #     accuracy_png_name_cle   = f'adversarial trained classifier {self._args.cla_model} accuracy on clean {self._args.dataset}'
            #     loss_png_name_cle       = f'adversarial trained classifier {self._args.cla_model} loss on clean {self._args.dataset}'               
            
            # SaveAccuracyCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_acc, global_adv_test_acc, accuracy_png_name_adv)
            # SaveLossCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_loss, global_adv_test_loss, loss_png_name_adv)
            # SaveAccuracyCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_acc, global_cle_test_acc, accuracy_png_name_cle)
            # SaveLossCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_loss, global_cle_test_loss, loss_png_name_cle)

    def __traintensorsetloop__(self):

        print("self._train_tensorset_x.shape:",self._train_tensorset_x.shape) 
        print("self._train_tensorset_y.shape:",self._train_tensorset_y.shape)       #   softlabel self._train_tensorset_y.shape: torch.Size([70000, 10])
        
        print("self._adv_test_tensorset_x.shape:",self._adv_test_tensorset_x.shape)
        print("self._adv_test_tensorset_y.shape:",self._adv_test_tensorset_y.shape) #   hard label self._adv_test_tensorset_y.shape: torch.Size([26032])

        print("self._cle_test_tensorset_x.shape:",self._cle_test_tensorset_x.shape)
        print("self._cle_test_tensorset_y.shape:",self._cle_test_tensorset_y.shape) #   hard label self._cle_test_tensorset_y.shape: torch.Size([26032])

        # print("self._train_tensorset_x[0]:",self._train_tensorset_x[0])
        # print("self._train_tensorset_y[0]:",self._train_tensorset_y[0])     
        # #   self._train_tensorset_y[0]: tensor([0.0000, 0.0000, 0.9842, 0.0000, 0.0000, 0.0158, 0.0000, 0.0000, 0.0000, 0.0000])
        # print("self._train_tensorset_x[10]:",self._train_tensorset_x[10])
        # print("self._train_tensorset_y[10]:",self._train_tensorset_y[10])        
        # #   self._train_tensorset_y[10]: tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8651, 0.1349, 0.0000, 0.0000, 0.0000])
        # raise error
        
        #  当前epoch分类模型在白盒对抗测试集上的准确率
        learned_model= self._model
        epoch_attack_classifier = AdvAttack(self._args, learned_model)    #   AdvAttack是MaggieClasssifier的子类
        target_model = epoch_attack_classifier.targetmodel()                #   即输入时的learned_model
        epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y) 
        epoch__adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(target_model,epoch_x_test_adv,epoch_y_test_adv)
        print(f'before mmat trained classifier accuary on adversarial testset:{epoch__adv_test_accuracy * 100:.4f}%' ) 
        print(f'before mmat trained classifier loss on adversarial testset:{epoch_adv_test_loss}' )    
        self._model = target_model

        trainset_len = len(self._train_tensorset_x)
        epoch_num = self._args.epochs                                               
        batchsize = self._args.batch_size
        batch_size = batchsize
        batch_num = int(np.ceil(trainset_len / float(batch_size)))

        shuffle_index = np.arange(trainset_len)
        shuffle_index = torch.tensor(shuffle_index)

        global_train_acc = []
        global_train_loss = []        
        global_adv_test_acc = []
        global_adv_test_loss = []
        global_cle_test_acc = []
        global_cle_test_loss = []

        for epoch_index in range (epoch_num):

            random.shuffle(shuffle_index)
            self.__adjustlearningrate__(epoch_index)     

            epoch_correct_num = 0
            epoch_total_loss = 0

            for batch_index in range (batch_num):

                x_trainbatch = self._train_tensorset_x[shuffle_index[batch_index * batch_size : (batch_index + 1) * batch_size]]
                y_trainbatch = self._train_tensorset_y[shuffle_index[batch_index * batch_size : (batch_index + 1) * batch_size]]                                                

                batch_imgs = x_trainbatch.cuda()
                batch_labs = y_trainbatch.cuda()

                self._optimizer.zero_grad()
                output = self._model(batch_imgs)

                #   计算损失
                lossfunction = 'ce'
                if lossfunction == 'mse':
                    softmax_outputs = torch.nn.functional.softmax(output, dim = 1)                              #   对每一行进行softmax
                    cla_loss = torch.nn.MSELoss()
                    batch_loss = cla_loss(softmax_outputs, batch_labs) 

                elif lossfunction == 'ce':
                    batch_loss = self.__CustomSoftlossFunction__(output, batch_labs)

                elif lossfunction == 'cosine':
                    softmax_outputs = torch.nn.functional.softmax(output, dim = 1)    
                    cla_loss = torch.cosine_similarity                                                          #   越接近1表明越相似
                    batch_loss = cla_loss(softmax_outputs, batch_labs) 
                    batch_loss = 1 - batch_loss         
                    batch_loss = batch_loss.mean()

                batch_loss.backward()
                self._optimizer.step()

                #   Top1 accuracy
                _, predicted_label_index = torch.max(output.data, 1)    

                _, batch_labs_maxindex = torch.max(batch_labs, 1)


                batch_correct_num = (predicted_label_index == batch_labs_maxindex).sum().item()     
                epoch_correct_num += batch_correct_num                                     
                
                epoch_total_loss += batch_loss
                print("[Epoch %d/%d] [Batch %d/%d] [Batch classify loss: %f] " % (epoch_index+1, epoch_num, batch_index+1, batch_num, 
                
                batch_loss.item()))
                
            #-------------------------------------------------------------------------    
            #   当前epoch分类模型在当前训练集epoch上的准确率    
            epoch_train_accuarcy = epoch_correct_num / trainset_len
            global_train_acc.append(epoch_train_accuarcy)                                   #   每个epoch训练完后的最新准确率list                  
            epoch_train_loss = epoch_total_loss / batch_num
            global_train_loss.append(epoch_train_loss)

            #   当前epoch分类模型在干净测试集上的准确率
            epoch_cle_test_accuracy, epoch_cle_test_loss = self.evaluatefromtensor(self._model, self._cle_test_tensorset_x, self._cle_test_tensorset_y)
            global_cle_test_acc.append(epoch_cle_test_accuracy)   
            global_cle_test_loss.append(epoch_cle_test_loss)
            print(f'{epoch_index+1:04d} epoch mmat trained classifier accuary on the clean testing examples:{epoch_cle_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index+1:04d} epoch mmat trained classifier loss on the clean testing examples:{epoch_cle_test_loss:.4f}' )   

            #  当前epoch分类模型在白盒对抗测试集上的准确率
            learned_model= self._model
            epoch_attack_classifier = AdvAttack(self._args, learned_model)    #   AdvAttack是MaggieClasssifier的子类
            target_model = epoch_attack_classifier.targetmodel()                #   即输入时的learned_model self._model

            epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y) 
            
            epoch_adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(target_model,epoch_x_test_adv,epoch_y_test_adv)
            global_adv_test_acc.append(epoch_adv_test_accuracy)   
            global_adv_test_loss.append(epoch_adv_test_loss)            
            print(f'mmat trained classifier accuary on adversarial testset:{epoch_adv_test_accuracy * 100:.4f}%' ) 
            print(f'mmat trained classifier loss on adversarial testset:{epoch_adv_test_loss}' )    

            self._model = target_model
            # raise error           

            #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-adv')
            os.makedirs(tensorboard_log_adv_acc_dir,exist_ok=True)    
            print("tensorboard_log_dir:",tensorboard_log_adv_acc_dir)   
            writer_adv_acc = SummaryWriter(log_dir = tensorboard_log_adv_acc_dir, comment= '-'+'advtestacc') 
            writer_adv_acc.add_scalar(tag = "epoch_adv_acc", scalar_value = epoch_adv_test_accuracy, global_step = epoch_index + 1 )
            writer_adv_acc.close()
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-adv')
            os.makedirs(tensorboard_log_adv_loss_dir,exist_ok=True)    
            writer_adv_loss = SummaryWriter(log_dir = tensorboard_log_adv_loss_dir, comment= '-'+'advtestloss') 
            writer_adv_loss.add_scalar(tag = "epoch_adv_loss", scalar_value = epoch_adv_test_loss, global_step = epoch_index + 1 )
            writer_adv_loss.close()
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-cle')
            os.makedirs(tensorboard_log_cle_acc_dir,exist_ok=True)    
            print("tensorboard_log_dir:",tensorboard_log_cle_acc_dir)   
            writer_cle_acc = SummaryWriter(log_dir = tensorboard_log_cle_acc_dir, comment= '-'+'cletestacc') 
            writer_cle_acc.add_scalar(tag = "epoch_cle_acc", scalar_value = epoch_cle_test_accuracy, global_step = epoch_index + 1 )
            writer_cle_acc.close()
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-cle')
            os.makedirs(tensorboard_log_cle_loss_dir,exist_ok=True)    
            writer_cle_loss = SummaryWriter(log_dir = tensorboard_log_cle_loss_dir, comment= '-'+'cletestloss') 
            writer_cle_loss.add_scalar(tag = "epoch_cle_loss", scalar_value = epoch_cle_test_loss, global_step = epoch_index + 1 )
            writer_cle_loss.close()
            #--------------------------------------------------

        return global_train_acc, global_adv_test_acc, global_cle_test_acc, global_train_loss, global_adv_test_loss, global_cle_test_loss

    def __evaluatesoftlabelfromtensor__(self, classifier, x_set:Tensor, y_set:Tensor):
        if torch.cuda.is_available():
            classifier.cuda()             

        eva_lossfunc = torch.nn.CrossEntropyLoss()

        batch_size = self._args.batch_size
        testset_total_num = len(x_set)
        batch_num = int( np.ceil( int(testset_total_num) / float(batch_size) ) )

        # print("x_set.shape:",x_set.shape)           #   x_set.shape: torch.Size([26032, 3, 32, 32])
        print("y_set.shape:",y_set.shape)           #   y_set.shape: torch.Size([26032])
        print("testset_total_num:",testset_total_num)       #   testset_total_num: 26032
        print("batch_num:",batch_num)                   #   batch_num: 813.5
        print("batch_size:",batch_size)       #  

        cla_model_name=self._args.cla_model
        print("cla_model_name:",cla_model_name)       #   cla_model_name: alexnet

        classify_loss = self._lossfunc
        epoch_correct_num = 0
        epoch_total_loss = 0

        for batch_index in range(batch_num):                                                #   进入batch迭代 共有num_batch个batch
            images = x_set[batch_index * batch_size : (batch_index + 1) * batch_size]
            labels = y_set[batch_index * batch_size : (batch_index + 1) * batch_size]                                                

            imgs = images.cuda()
            labs = labels.cuda()

            with torch.no_grad():

                if cla_model_name == 'inception_v3':
                    output, aux = classifier(imgs)
                
                elif cla_model_name == 'googlenet':
                    if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'svhn':  #   只有imagenet和svhn搭配googlenet时是返回一个值
                        output = classifier(imgs)
                    else:
                        output, aux1, aux2 = classifier(imgs)
                else:
                    output = classifier(imgs)         

                loss = eva_lossfunc(output,labs)
                _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
                
                batch_same_num = (predicted_label_index == labs).sum().item()
                epoch_correct_num += batch_same_num
                epoch_total_loss += loss

        test_accuracy = epoch_correct_num / testset_total_num
        test_loss = epoch_total_loss / batch_num                  

        return test_accuracy, test_loss

    def __CustomSoftlossFunction__(self, batch_outputs, o_batch):       
        #   求第一大概率标签的混合系数 alpha_1
        alpha_1, w1_label_index = torch.max(o_batch, 1)                            

        #   求第二大概率标签的混合系数 alpha_2
        modified_mixed_label = copy.deepcopy(o_batch)

        alpha_2 = []
        w2_label_index = []
        for i in range(len(o_batch)):

            modified_mixed_label[i][w1_label_index[i]] = 0        
            
            if torch.nonzero(modified_mixed_label[i]).size(0) == 0:
                # print("混合label的最大值维度置零后，其他全为0！")
                ind = w1_label_index[i].unsqueeze(0).cuda() #   第二大label设置为和第一大label一样
                val = torch.zeros(1, dtype = torch.float32).cuda()
                alpha_2.append(val)
                w2_label_index.append(ind)
                # raise error
                            
            else:
                # print("混合label的最大值维度置零后，其他不全为0！")
                mix_label = modified_mixed_label[i]
                mix_label = mix_label.unsqueeze(0)
                val, ind = torch.max(mix_label, 1)
                alpha_2.append(val)
                w2_label_index.append(ind)

        w2_label_index = torch.cat(w2_label_index) 
        alpha_2 = torch.cat(alpha_2) 

        # print("batch_outputs[0]:",batch_outputs[0])
        # print("alpha_1[0]:",alpha_1[0])
        # print("alpha_2[0]:",alpha_2[0])
        # print("w1_label_index[0]:",w1_label_index[0])
        # print("w2_label_index[0]:",w2_label_index[0])

        cla_loss =  torch.nn.CrossEntropyLoss(reduction = 'none').cuda()
        loss_a = cla_loss(batch_outputs, w1_label_index)
        loss_b = cla_loss(batch_outputs, w2_label_index)
        loss = alpha_1 * loss_a + alpha_2 * loss_b
        loss = loss.mean()

        return loss

    #   representation mixup training
    def rmt(self, args,cle_w_train,cle_y_train, cle_train_dataloader, cle_x_test, cle_y_test, adv_x_test,adv_y_test,exp_result_dir,stylegan2ada_config_kwargs):

        print("cle_w_train.shape:",cle_w_train.shape)   
        print("cle_y_train.shape:",cle_y_train.shape)

        print("cle_x_test.shape:",cle_x_test.shape)
        print("cle_y_test.shape:",cle_y_test.shape)        

        print("adv_x_test.shape:",adv_x_test.shape)
        print("adv_y_test.shape:",adv_y_test.shape) 

        """
        cle_w_train.shape: torch.Size([11339, 8, 512])
        cle_y_train.shape: torch.Size([11339, 8, 10])
        cle_x_test.shape: torch.Size([26032, 3, 32, 32])
        cle_y_test.shape: torch.Size([26032])
        adv_x_test.shape: torch.Size([26032, 3, 32, 32])
        adv_y_test.shape: torch.Size([26032])
        """  

        self._exp_result_dir = exp_result_dir
        if self._args.defense_mode == "rmt":
            self._exp_result_dir = os.path.join(self._exp_result_dir,f'rmt-{self._args.dataset}-dataset')
        os.makedirs(self._exp_result_dir,exist_ok=True)            
        if torch.cuda.is_available():
            self._lossfunc.cuda()
            self._model.cuda()          #   self._model在初始化时被赋值为了读入的模型

        self._train_tensorset_x = cle_w_train
        self._train_tensorset_y = cle_y_train

        self._adv_test_tensorset_x = adv_x_test
        self._adv_test_tensorset_y = adv_y_test

        self._cle_test_tensorset_x = cle_x_test
        self._cle_test_tensorset_y = cle_y_test

        self._train_dataloader = cle_train_dataloader

        if args.whitebox == True:
            #   当前分类模型在白盒对抗测试集上的准确率
            epoch_attack_classifier = AdvAttack(args = self._args, learned_model= self.model())              #   AdvAttack是MaggieClasssifier的子类
            self._model = epoch_attack_classifier.targetmodel()
            epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y) 
        elif args.blackbox == True:
            #   当前分类模型在黑盒对抗测试集上的准确率
            epoch_x_test_adv = self._adv_test_tensorset_x
            epoch_y_test_adv = self._adv_test_tensorset_y        
        
        epoch__adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(self._model, epoch_x_test_adv,epoch_y_test_adv)

        print(f'Accuary of before rmt trained classifier on adversarial testset:{epoch__adv_test_accuracy * 100:.4f}%' ) 
        print(f'Loss of before mmat trained classifier on adversarial testset:{epoch_adv_test_loss}' )    

        #----------train----
        w_trainset_len = len(self._train_tensorset_x)
        batch_size = self._args.batch_size
        w_batch_num = int(np.ceil(w_trainset_len / float(batch_size)))

        print("w_trainset_len:",w_trainset_len)
        print("batch_size:",batch_size)
        print("w_batch_num:",w_batch_num)

        shuffle_index = np.arange(w_trainset_len)
        shuffle_index = torch.tensor(shuffle_index)

        for epoch_index in range(self._args.epochs):
            print("\n")
            random.shuffle(shuffle_index)
            self.__adjustlearningrate__(epoch_index)       

            epoch_total_loss = 0

            for batch_index, (raw_img_batch, raw_lab_batch) in enumerate(self._train_dataloader):         #   加载原始训练集batch
                
                #--------------raw batch training--------------
                # softlabel
                # print("raw_lab_batch:",raw_lab_batch)
                # print("raw_lab_batch.type:",type(raw_lab_batch))

                """
                raw_lab_batch: tensor([9, 9, 0, 6])
                raw_lab_batch.type: <class 'torch.Tensor'>
                """

                raw_lab_batch = LongTensor(raw_lab_batch)                           #   list型转为tensor
                raw_lab_batch = torch.nn.functional.one_hot(raw_lab_batch, args.n_classes).float()
                
                #-----------maggie cat clean and mix------------
                if (batch_index + 1) % w_batch_num == 0:
                    right_index = w_trainset_len
                else:
                    right_index = ( (batch_index + 1) % w_batch_num ) * batch_size
                # print("right_index:",right_index) 

                pro_img_batch = self._train_tensorset_x[shuffle_index[(batch_index % w_batch_num) * batch_size : right_index]]
                pro_lab_batch = self._train_tensorset_y[shuffle_index[(batch_index % w_batch_num) * batch_size : right_index]]                   
                
                # print("20211026 mixup test")
                mix_img_batch, mix_lab_batch = mixup_data(args, exp_result_dir, stylegan2ada_config_kwargs, pro_img_batch, pro_lab_batch)      #   混合样本 two-hot标签
                
                aug_x_train = torch.cat([raw_img_batch, mix_img_batch], dim=0)
                aug_y_train = torch.cat([raw_lab_batch, mix_lab_batch], dim=0)

                inputs = aug_x_train.cuda()
                targets = aug_y_train.cuda()
                
                self._optimizer.zero_grad()
                self._model.train()                 #   train mode

                # outputs = self._model(inputs)
                if self._args.cla_model == 'inception_v3':
                    outputs, aux = self._model(inputs)
                elif self._args.cla_model == 'googlenet':
                    outputs, aux1, aux2 = self._model(inputs)
                else:
                    outputs = self._model(inputs)

               #   计算损失
                lossfunction = 'ce'
                if lossfunction == 'ce':
                    loss = self.__CustomSoftlossFunction__(outputs, targets)
                elif lossfunction == 'mse':
                    softmax_outputs = torch.nn.functional.softmax(outputs, dim = 1)                              #   对每一行进行softmax
                    cla_lossfun = torch.nn.MSELoss().cuda()
                    loss = cla_lossfun(softmax_outputs, targets) 
                elif lossfunction == 'cosine':
                    softmax_outputs = torch.nn.functional.softmax(outputs, dim = 1)    
                    cla_lossfun = torch.cosine_similarity                                                          #   越接近1表明越相似
                    loss = cla_lossfun(softmax_outputs, targets) 
                    loss = 1 - loss         
                    loss =loss.mean()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_total_loss += loss
                #------------------------------
                print("[Epoch %d/%d] [Batch %d/%d] [Batch classify loss: %f]" % (epoch_index+1, self._args.epochs, batch_index+1, len(self._train_dataloader), loss.item()))
            #   finish batch training
            
            #   当前epoch分类模型在干净测试集上的准确率
            epoch_cle_test_accuracy, epoch_cle_test_loss = self.evaluatefromtensor(self._model, self._cle_test_tensorset_x, self._cle_test_tensorset_y)
            # global_cle_test_acc.append(epoch_cle_test_accuracy)   
            # global_cle_test_loss.append(epoch_cle_test_loss)
            print(f'{epoch_index+1:04d} epoch rmt trained classifier accuary on the clean testing examples:{epoch_cle_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index+1:04d} epoch rmt trained classifier loss on the clean testing examples:{epoch_cle_test_loss:.4f}' )   

            if args.whitebox == True:
                #  当前epoch分类模型在白盒对抗测试集上的准确率
                epoch_attack_classifier = AdvAttack(self._args, self._model)    #   AdvAttack是MaggieClasssifier的子类
                self._model = epoch_attack_classifier.targetmodel()                #   即输入时的learned_model self._model
                epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y)         
            elif args.blackbox == True:
                 #当前epoch分类模型在黑盒对抗测试集上的准确率
                epoch_x_test_adv = self._adv_test_tensorset_x
                epoch_y_test_adv = self._adv_test_tensorset_y

            epoch_adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(self._model,epoch_x_test_adv,epoch_y_test_adv)
            # global_adv_test_acc.append(epoch_adv_test_accuracy)   
            # global_adv_test_loss.append(epoch_adv_test_loss)                 
            print(f'{epoch_index+1:04d} epoch rmt trained classifier accuary on adversarial testset:{epoch_adv_test_accuracy * 100:.4f}%' ) 
            print(f'{epoch_index+1:04d} epoch rmt trained classifier loss on adversarial testset:{epoch_adv_test_loss}' )    


            # if (epoch_index+1) % 11== 0 and epoch_index > 0:
            # if epoch_adv_test_accuracy  >= 0.2:
            #     torch.save(self._model,f'{self._exp_result_dir}/rmt-trained-classifier-{self._args.cla_model}-on-clean-{self._args.dataset}-epoch-{epoch_index+1:04d}.pkl')            

            #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-adv')
            os.makedirs(tensorboard_log_adv_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_adv_acc_dir)   
            writer_adv_acc = SummaryWriter(log_dir = tensorboard_log_adv_acc_dir, comment= '-'+'advtestacc') 
            writer_adv_acc.add_scalar(tag = "epoch_adv_acc", scalar_value = epoch_adv_test_accuracy, global_step = epoch_index + 1 )
            writer_adv_acc.close()
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-adv')
            os.makedirs(tensorboard_log_adv_loss_dir,exist_ok=True)    
            writer_adv_loss = SummaryWriter(log_dir = tensorboard_log_adv_loss_dir, comment= '-'+'advtestloss') 
            writer_adv_loss.add_scalar(tag = "epoch_adv_loss", scalar_value = epoch_adv_test_loss, global_step = epoch_index + 1 )
            writer_adv_loss.close()
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-cle')
            os.makedirs(tensorboard_log_cle_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_cle_acc_dir)   
            writer_cle_acc = SummaryWriter(log_dir = tensorboard_log_cle_acc_dir, comment= '-'+'cletestacc') 
            writer_cle_acc.add_scalar(tag = "epoch_cle_acc", scalar_value = epoch_cle_test_accuracy, global_step = epoch_index + 1 )
            writer_cle_acc.close()
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-cle')
            os.makedirs(tensorboard_log_cle_loss_dir,exist_ok=True)    
            writer_cle_loss = SummaryWriter(log_dir = tensorboard_log_cle_loss_dir, comment= '-'+'cletestloss') 
            writer_cle_loss.add_scalar(tag = "epoch_cle_loss", scalar_value = epoch_cle_test_loss, global_step = epoch_index + 1 )
            writer_cle_loss.close()
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_train_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-train')
            os.makedirs(tensorboard_log_train_loss_dir,exist_ok=True)    
            writer_tra_loss = SummaryWriter(log_dir = tensorboard_log_train_loss_dir, comment= '-'+'augtrainloss') 
            writer_tra_loss.add_scalar(tag = "epoch_augtrain_loss", scalar_value = epoch_total_loss/len(self._train_dataloader), global_step = epoch_index + 1 )
            writer_tra_loss.close()
            #--------------------------------------------------

    def __adjustlearningrate__(self, epoch_index):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""#   每隔10epoch除以一次10

        if self._args.train_mode == 'cla-train':

            if self._args.dataset == 'cifar10':
                if self._args.cla_model == 'resnet34':     
                    # lr = self._args.lr * (0.1 ** (epoch_index // 10))                   #   每隔10epoch除以一次10

                    if epoch_index <= 7:
                        lr = self._args.lr                                  #   0.01
                    elif epoch_index >= 8:
                        lr = self._args.lr * 0.1                            #   0.001

                elif self._args.cla_model == 'alexnet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))              

                    # if epoch_index <= 8:
                    #     lr = self._args.lr                                  #   0.01
                    # elif epoch_index >= 9:
                    #     lr = self._args.lr * 0.1                            #   0.001

                elif self._args.cla_model =='resnet18':
                    # lr = self._args.lr * (0.1 ** (epoch_index // 10))

                    if epoch_index <= 5:
                        lr = self._args.lr                                  #   0.01
                    elif epoch_index >= 6 and epoch_index <= 7:
                        lr = self._args.lr * 0.1                            #   0.001
                    elif epoch_index >= 8:
                        lr = self._args.lr * 0.01                            #   0.0001

                elif self._args.cla_model =='resnet50':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='vgg19':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='inception_v3':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='densenet169':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='googlenet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='preactresnet18':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='preactresnet34':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='preactresnet50':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

            elif self._args.dataset == 'svhn':
                if self._args.cla_model == 'resnet34':     
                    # lr = self._args.lr * (0.1 ** (epoch_index // 10))                   #   每隔10epoch除以一次10

                    if epoch_index <= 7:
                        lr = self._args.lr                                  #   0.01
                    elif epoch_index >= 8:
                        lr = self._args.lr * 0.1                            #   0.001

                elif self._args.cla_model == 'alexnet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))              

                    # if epoch_index <= 8:
                    #     lr = self._args.lr                                  #   0.01
                    # elif epoch_index >= 9:
                    #     lr = self._args.lr * 0.1                            #   0.001

                elif self._args.cla_model =='resnet18':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                    # if epoch_index <= 5:
                    #     lr = self._args.lr                                  #   0.01
                    # elif epoch_index >= 6 and epoch_index <= 7:
                    #     lr = self._args.lr * 0.1                            #   0.001
                    # elif epoch_index >= 8:
                    #     lr = self._args.lr * 0.01                            #   0.0001

                elif self._args.cla_model =='resnet50':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='vgg19':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='inception_v3':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='densenet169':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='googlenet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

            elif self._args.dataset == 'imagenetmixed10':
                if self._args.cla_model == 'alexnet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 20))  

                elif self._args.cla_model == 'vgg19':
                    lr = self._args.lr * (0.1 ** (epoch_index // 20))  

            elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
                if self._args.cla_model == 'alexnet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model == 'resnet18':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model == 'resnet34':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model == 'resnet50':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model =='vgg19':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='inception_v3':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='densenet169':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='googlenet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

            elif self._args.dataset == 'cifar100':
                if self._args.cla_model == 'resnet34':     
                    # lr = self._args.lr * (0.1 ** (epoch_index // 10))                   #   每隔10epoch除以一次10

                    if epoch_index <= 7:
                        lr = self._args.lr                                  #   0.01
                    elif epoch_index >= 8:
                        lr = self._args.lr * 0.1                            #   0.001
                    elif epoch_index >= 15:
                        lr = self._args.lr * 0.01                            #   0.0001

                elif self._args.cla_model == 'alexnet':
                    # lr = self._args.lr * (0.1 ** (epoch_index // 10))              

                    if epoch_index <= 11:
                        lr = self._args.lr                                  #   0.01
                    elif epoch_index >= 12:
                        lr = self._args.lr * 0.1                            #   0.001


                elif self._args.cla_model =='resnet18':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                    # if epoch_index <= 5:
                    #     lr = self._args.lr                                  #   0.01
                    # elif epoch_index >= 6 and epoch_index <= 7:
                    #     lr = self._args.lr * 0.1                            #   0.001
                    # elif epoch_index >= 8:
                    #     lr = self._args.lr * 0.01                            #   0.0001

                elif self._args.cla_model =='resnet50':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='vgg19':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='inception_v3':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='densenet169':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='googlenet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

            elif self._args.dataset == 'stl10':
                if self._args.cla_model == 'alexnet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model == 'resnet18':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model == 'resnet34':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model == 'resnet50':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))  
                
                elif self._args.cla_model =='vgg19':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                
                elif self._args.cla_model =='inception_v3':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='densenet169':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                elif self._args.cla_model =='googlenet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
            
        elif self._args.defense_mode == 'mmat':
            if self._args.dataset == 'svhn':
                if self._args.cla_model == 'alexnet':
                    
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

                    # if epoch_index <= 3:
                    #     lr = self._args.lr                                  #   0.1
                    # elif epoch_index >= 4 and epoch_index <= 5:
                    #     lr = self._args.lr * 0.1                            #   0.01 
                    # elif epoch_index >= 6 and epoch_index <= 10:
                    #     lr = self._args.lr * 0.1                            #   0.001   
                    # elif epoch_index >= 11:
                    #     lr = self._args.lr * 0.1                            #   0.0001   


                    # if epoch_index <= 3:
                    #     lr = 0.1 + (0.1 * epoch_index)  # 0.1 -> 0.2 -> 0.3 -> 0.4 
                    # elif epoch_index >= 4 and epoch_index <= 10:
                    #     lr = 0.01
                    # elif epoch_index >= 11 and epoch_index <= 20:
                    #     lr = 0.001
                elif self._args.cla_model == 'resnet18':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

            if self._args.dataset == 'cifar10':
                if self._args.cla_model == 'alexnet':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))
                elif self._args.cla_model == 'resnet18':
                    lr = self._args.lr * (0.1 ** (epoch_index // 10))

        elif self._args.defense_mode == 'rmt':
            if self._args.dataset == 'svhn' or self._args.dataset == 'cifar10' or self._args.dataset == 'kmnist':

                lr = self._args.lr * (0.1 ** (epoch_index // 10))    

        elif self._args.defense_mode == 'inputmixup':
            if self._args.dataset == 'svhn' or self._args.dataset == 'cifar10' or self._args.dataset == 'kmnist':

                lr = self._args.lr * (0.1 ** (epoch_index // 10))    

        elif self._args.defense_mode == 'at':
            if self._args.dataset == 'svhn' or self._args.dataset == 'cifar10' or self._args.dataset == 'kmnist':

                lr = self._args.lr * (0.1 ** (epoch_index // 10))    

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


        print(f'{epoch_index}epoch learning rate:{lr}')             #   0epoch learning rate:0.01

    # 20211107 input mixup train
    def inputmixuptrain(self,args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir):
        print("compare with---------input mixup train--------------")

        print("cle_x_train.shape:",cle_x_train.shape)   
        print("cle_y_train.shape:",cle_y_train.shape)

        print("cle_x_test.shape:",cle_x_test.shape)
        print("cle_y_test.shape:",cle_y_test.shape)        

        print("adv_x_test.shape:",adv_x_test.shape)
        print("adv_y_test.shape:",adv_y_test.shape) 

        """
        cle_x_test.shape: torch.Size([10000, 3, 32, 32])
        cle_y_test.shape: torch.Size([10000])
        adv_x_test.shape: torch.Size([10000, 3, 32, 32])
        adv_y_test.shape: torch.Size([10000])
        """

        self._exp_result_dir = exp_result_dir
        if self._args.defense_mode == "inputmixup":
            self._exp_result_dir = os.path.join(self._exp_result_dir,f'rmt-{self._args.dataset}-dataset')
        os.makedirs(self._exp_result_dir,exist_ok=True) 

        if torch.cuda.is_available():
            self._lossfunc.cuda()
            self._model.cuda()          #   self._model在初始化时被赋值为了读入的模型

        self._train_tensorset_x = cle_x_train
        self._train_tensorset_y = cle_y_train

        self._adv_test_tensorset_x = adv_x_test
        self._adv_test_tensorset_y = adv_y_test

        self._cle_test_tensorset_x = cle_x_test
        self._cle_test_tensorset_y = cle_y_test

        self._train_dataloader = cle_train_dataloader

        #   获取对抗样本
        if args.whitebox == True:
            print("白盒")
            #   当前分类模型在白盒对抗测试集上的准确率 现场生成
            epoch_attack_classifier = AdvAttack(args = self._args, learned_model= self.model())              #   AdvAttack是MaggieClasssifier的子类
            self._model = epoch_attack_classifier.targetmodel()
            epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y) 
        elif args.blackbox == True:
            print("黑盒")
            #   当前分类模型在黑盒对抗测试集上的准确率 传入
            epoch_x_test_adv = self._adv_test_tensorset_x
            epoch_y_test_adv = self._adv_test_tensorset_y     

        epoch__adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(self._model, epoch_x_test_adv,epoch_y_test_adv)
        print(f'Accuary of before rmt trained classifier on adversarial testset:{epoch__adv_test_accuracy * 100:.4f}%' ) 
        print(f'Loss of before mmat trained classifier on adversarial testset:{epoch_adv_test_loss}' )    
        
        #----------train----
        w_trainset_len = len(self._train_tensorset_x)
        batch_size = self._args.batch_size
        w_batch_num = int(np.ceil(w_trainset_len / float(batch_size)))

        print("w_trainset_len:",w_trainset_len)
        print("batch_size:",batch_size)
        print("w_batch_num:",w_batch_num)
        """
        w_trainset_len: 25397
        batch_size: 256
        w_batch_num: 100
        """
        shuffle_index = np.arange(w_trainset_len)
        shuffle_index = torch.tensor(shuffle_index)

        for epoch_index in range(self._args.epochs):
            print("\n")
            random.shuffle(shuffle_index)
            self.__adjustlearningrate__(epoch_index)       

            epoch_total_loss = 0

            for batch_index, (raw_img_batch, raw_lab_batch) in enumerate(self._train_dataloader):         #   加载原始训练集batch
                raw_lab_batch = LongTensor(raw_lab_batch)                           #   list型转为tensor
                raw_lab_batch = torch.nn.functional.one_hot(raw_lab_batch, args.n_classes).float()
                
                # print("raw_img_batch.shape:",raw_img_batch.shape)
                # print("raw_lab_batch.shape:",raw_lab_batch.shape)
                # """
                # raw_img_batch.shape: torch.Size([256, 3, 32, 32])
                # raw_lab_batch.shape: torch.Size([256, 10])
                # """


                #-----------maggie cat clean and mix------------
                if (batch_index + 1) % w_batch_num == 0:
                    right_index = w_trainset_len
                else:
                    right_index = ( (batch_index + 1) % w_batch_num ) * batch_size
                # print("right_index:",right_index) 

                cle_img_batch = self._train_tensorset_x[shuffle_index[(batch_index % w_batch_num) * batch_size : right_index]]
                cle_lab_batch = self._train_tensorset_y[shuffle_index[(batch_index % w_batch_num) * batch_size : right_index]]                   
                
                # print("20211107 mixup test")
                mix_img_batch, mix_lab_batch = input_mixup_data(args, cle_img_batch, cle_lab_batch)      #   混合样本 two-hot标签              
                
                #------------------20211108--------------------
                # aug_x_train = torch.cat([raw_img_batch, mix_img_batch], dim=0)
                # aug_y_train = torch.cat([raw_lab_batch, mix_lab_batch], dim=0)
                aug_x_train = mix_img_batch
                aug_y_train = mix_lab_batch

                # if self._args.defense_mode == 'inputmixup':
                #     #   在inputmixup中不混合原样本
                #     aug_x_train = mix_img_batch
                #     aug_y_train = mix_lab_batch
                #     # aug_x_train = raw_img_batch #   纯粹用干净样本微调是否有效
                #     # aug_y_train = raw_lab_batch                    
                # else:
                #     aug_x_train = torch.cat([raw_img_batch, mix_img_batch], dim=0)
                #     aug_y_train = torch.cat([raw_lab_batch, mix_lab_batch], dim=0)
                #---------------------------------------------

                inputs = aug_x_train.cuda()
                targets = aug_y_train.cuda()
                
                # self._optimizer.zero_grad()
                self._model.train()                 #   train mode

                # outputs = self._model(inputs)
                if self._args.cla_model == 'inception_v3':
                    outputs, aux = self._model(inputs)
                elif self._args.cla_model == 'googlenet':
                    outputs, aux1, aux2 = self._model(inputs)
                else:
                    outputs = self._model(inputs)

               #   计算损失
                lossfunction = 'ce'
                if lossfunction == 'ce':
                    loss = self.__CustomSoftlossFunction__(outputs, targets)
                elif lossfunction == 'mse':
                    softmax_outputs = torch.nn.functional.softmax(outputs, dim = 1)                              #   对每一行进行softmax
                    cla_lossfun = torch.nn.MSELoss().cuda()
                    loss = cla_lossfun(softmax_outputs, targets) 
                elif lossfunction == 'cosine':
                    softmax_outputs = torch.nn.functional.softmax(outputs, dim = 1)    
                    cla_lossfun = torch.cosine_similarity                                                          #   越接近1表明越相似
                    loss = cla_lossfun(softmax_outputs, targets) 
                    loss = 1 - loss         
                    loss =loss.mean()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_total_loss += loss
                #------------------------------
                print("[Epoch %d/%d] [Batch %d/%d] [Batch classify loss: %f]" % (epoch_index+1, self._args.epochs, batch_index+1, len(self._train_dataloader), loss.item()))
            #   finish batch training
                
            
            #   当前epoch分类模型在干净测试集上的准确率
            epoch_cle_test_accuracy, epoch_cle_test_loss = self.evaluatefromtensor(self._model, self._cle_test_tensorset_x, self._cle_test_tensorset_y)
            # global_cle_test_acc.append(epoch_cle_test_accuracy)   
            # global_cle_test_loss.append(epoch_cle_test_loss)
            print(f'{epoch_index+1:04d} epoch rmt trained classifier accuary on the clean testing examples:{epoch_cle_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index+1:04d} epoch rmt trained classifier loss on the clean testing examples:{epoch_cle_test_loss:.4f}' )   

            if args.whitebox == True:
                #  当前epoch分类模型在白盒对抗测试集上的准确率
                epoch_attack_classifier = AdvAttack(self._args, self._model)    #   AdvAttack是MaggieClasssifier的子类
                self._model = epoch_attack_classifier.targetmodel()                #   即输入时的learned_model self._model
                epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y)         
            elif args.blackbox == True:
                 #当前epoch分类模型在黑盒对抗测试集上的准确率
                epoch_x_test_adv = self._adv_test_tensorset_x
                epoch_y_test_adv = self._adv_test_tensorset_y

            epoch_adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(self._model,epoch_x_test_adv,epoch_y_test_adv)               
            print(f'{epoch_index+1:04d} epoch rmt trained classifier accuary on adversarial testset:{epoch_adv_test_accuracy * 100:.4f}%' ) 
            print(f'{epoch_index+1:04d} epoch rmt trained classifier loss on adversarial testset:{epoch_adv_test_loss}' )    


            #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-adv')
            os.makedirs(tensorboard_log_adv_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_adv_acc_dir)   
            writer_adv_acc = SummaryWriter(log_dir = tensorboard_log_adv_acc_dir, comment= '-'+'advtestacc') 
            writer_adv_acc.add_scalar(tag = "epoch_adv_acc", scalar_value = epoch_adv_test_accuracy, global_step = epoch_index + 1 )
            writer_adv_acc.close()
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-adv')
            os.makedirs(tensorboard_log_adv_loss_dir,exist_ok=True)    
            writer_adv_loss = SummaryWriter(log_dir = tensorboard_log_adv_loss_dir, comment= '-'+'advtestloss') 
            writer_adv_loss.add_scalar(tag = "epoch_adv_loss", scalar_value = epoch_adv_test_loss, global_step = epoch_index + 1 )
            writer_adv_loss.close()
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-cle')
            os.makedirs(tensorboard_log_cle_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_cle_acc_dir)   
            writer_cle_acc = SummaryWriter(log_dir = tensorboard_log_cle_acc_dir, comment= '-'+'cletestacc') 
            writer_cle_acc.add_scalar(tag = "epoch_cle_acc", scalar_value = epoch_cle_test_accuracy, global_step = epoch_index + 1 )
            writer_cle_acc.close()
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-cle')
            os.makedirs(tensorboard_log_cle_loss_dir,exist_ok=True)    
            writer_cle_loss = SummaryWriter(log_dir = tensorboard_log_cle_loss_dir, comment= '-'+'cletestloss') 
            writer_cle_loss.add_scalar(tag = "epoch_cle_loss", scalar_value = epoch_cle_test_loss, global_step = epoch_index + 1 )
            writer_cle_loss.close()
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_train_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-train')
            os.makedirs(tensorboard_log_train_loss_dir,exist_ok=True)    
            writer_tra_loss = SummaryWriter(log_dir = tensorboard_log_train_loss_dir, comment= '-'+'augtrainloss') 
            writer_tra_loss.add_scalar(tag = "epoch_augtrain_loss", scalar_value = epoch_total_loss/len(self._train_dataloader), global_step = epoch_index + 1 )
            writer_tra_loss.close()
            #--------------------------------------------------
          

    def advtrain(self, args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir):
        print("compare with---------adversarial train--------------")

        # print("adv_x_train.shape:",adv_x_train.shape)   
        # print("adv_y_train.shape:",adv_y_train.shape)

        # print("cle_x_test.shape:",cle_x_test.shape)
        # print("cle_y_test.shape:",cle_y_test.shape)        

        # print("adv_x_test.shape:",adv_x_test.shape)
        # print("adv_y_test.shape:",adv_y_test.shape) 

        """
        adv_x_train.shape: torch.Size([50000, 3, 32, 32])
        adv_y_train.shape: torch.Size([50000])
        cle_x_test.shape: torch.Size([10000, 3, 32, 32])
        cle_y_test.shape: torch.Size([10000])
        adv_x_test.shape: torch.Size([10000, 3, 32, 32])
        adv_y_test.shape: torch.Size([10000])        
        """

        self._exp_result_dir = exp_result_dir
        self._lossfunc = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self._lossfunc.cuda()
            self._model.cuda()          #   self._model在初始化时被赋值为了读入的模型

        self._train_tensorset_x = adv_x_train
        self._train_tensorset_y = adv_y_train

        self._adv_test_tensorset_x = adv_x_test
        self._adv_test_tensorset_y = adv_y_test

        self._cle_test_tensorset_x = cle_x_test
        self._cle_test_tensorset_y = cle_y_test

        self._train_dataloader = cle_train_dataloader

        #   获取对抗样本
        if args.whitebox == True:
            print("白盒")
            #   当前分类模型在白盒对抗测试集上的准确率 现场生成
            epoch_attack_classifier = AdvAttack(args = self._args, learned_model= self.model())              #   AdvAttack是MaggieClasssifier的子类
            self._model = epoch_attack_classifier.targetmodel()
            epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y) 
        elif args.blackbox == True:
            #   当前分类模型在黑盒对抗测试集上的准确率 传入
            print("黑盒")
            epoch_x_test_adv = self._adv_test_tensorset_x # omfgsm黑盒
            epoch_y_test_adv = self._adv_test_tensorset_y     

        epoch_adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(self._model, epoch_x_test_adv,epoch_y_test_adv)
        print(f'Accuary of before rmt trained classifier on adversarial testset:{epoch_adv_test_accuracy * 100:.4f}%' ) 
        print(f'Loss of before mmat trained classifier on adversarial testset:{epoch_adv_test_loss}' )    

        #----------train----
        adv_trainset_len = len(self._train_tensorset_x)
        batch_size = self._args.batch_size
        adv_batch_num = int(np.ceil(adv_trainset_len / float(batch_size)))

        print("adv_trainset_len:",adv_trainset_len) #5w
        print("batch_size:",batch_size)
        print("adv_batch_num:",adv_batch_num)

        shuffle_index = np.arange(adv_trainset_len)
        shuffle_index = torch.tensor(shuffle_index)

        #----------train----
        for epoch_index in range(self._args.epochs):
            print("\n")
            random.shuffle(shuffle_index)
            self.__adjustlearningrate__(epoch_index)     
            epoch_total_loss = 0

            for batch_index, (raw_img_batch, raw_lab_batch) in enumerate(self._train_dataloader):         #   加载原始训练集batch
                
                # print("raw_img_batch.shape:",raw_img_batch.shape)
                # print("raw_lab_batch.shape:",raw_lab_batch.shape)
                # """
                # raw_img_batch.shape: torch.Size([256, 3, 32, 32])
                # raw_lab_batch.shape: torch.Size([256])
                # """
               #-----------maggie cat clean and mix------------
                if (batch_index + 1) % adv_batch_num == 0:
                    right_index = adv_trainset_len
                else:
                    right_index = ( (batch_index + 1) % adv_batch_num ) * batch_size
                # print("right_index:",right_index) 

                adv_img_batch = self._train_tensorset_x[shuffle_index[(batch_index % adv_batch_num) * batch_size : right_index]]
                adv_lab_batch = self._train_tensorset_y[shuffle_index[(batch_index % adv_batch_num) * batch_size : right_index]]                   
                # print("adv_img_batch.shape:",adv_img_batch.shape)
                # print("adv_lab_batch.shape:",adv_lab_batch.shape)                
                # """
                # adv_img_batch.shape: torch.Size([256, 3, 32, 32])
                # adv_lab_batch.shape: torch.Size([256])
                # """
                #------------------20211110--------------------

                aug_x_train = torch.cat([raw_img_batch, adv_img_batch], dim=0)
                aug_y_train = torch.cat([raw_lab_batch, adv_lab_batch], dim=0)                
                # aug_x_train = raw_img_batch
                # aug_y_train = raw_lab_batch
                # aug_x_train = adv_img_batch
                # aug_y_train = adv_lab_batch
                # raise error


                inputs = aug_x_train.cuda()
                targets = aug_y_train.cuda()
                
                # self._optimizer.zero_grad()
                self._model.train()                 #   train mode

                if self._args.cla_model == 'inception_v3':
                    outputs, aux = self._model(inputs)
                elif self._args.cla_model == 'googlenet':
                    outputs, aux1, aux2 = self._model(inputs)
                else:
                    outputs = self._model(inputs)

                loss = self._lossfunc(outputs, targets)
                
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_total_loss += loss
                #------------------------------
                print("[Epoch %d/%d] [Batch %d/%d] [Batch classify loss: %f]" % (epoch_index+1, self._args.epochs, batch_index+1, len(self._train_dataloader), loss.item()))
            #   finish batch training
                
            # self._model.
            #   当前epoch分类模型在干净测试集上的准确率
            epoch_cle_test_accuracy, epoch_cle_test_loss = self.evaluatefromtensor(self._model, self._cle_test_tensorset_x, self._cle_test_tensorset_y)
            # global_cle_test_acc.append(epoch_cle_test_accuracy)   
            # global_cle_test_loss.append(epoch_cle_test_loss)
            print(f'{epoch_index+1:04d} epoch at trained classifier accuary on the clean testing examples:{epoch_cle_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index+1:04d} epoch at trained classifier loss on the clean testing examples:{epoch_cle_test_loss:.4f}' )   

            if args.whitebox == True:
                #  当前epoch分类模型在白盒对抗测试集上的准确率
                epoch_attack_classifier = AdvAttack(self._args, self._model)    #   AdvAttack是MaggieClasssifier的子类
                self._model = epoch_attack_classifier.targetmodel()                #   即输入时的learned_model self._model
                epoch_x_test_adv, epoch_y_test_adv = epoch_attack_classifier.generateadvfromtestsettensor(self._cle_test_tensorset_x, self._cle_test_tensorset_y)         
            elif args.blackbox == True:
                 #当前epoch分类模型在黑盒对抗测试集上的准确率
                epoch_x_test_adv = self._adv_test_tensorset_x
                epoch_y_test_adv = self._adv_test_tensorset_y

            epoch_adv_test_accuracy, epoch_adv_test_loss = self.evaluatefromtensor(self._model,epoch_x_test_adv,epoch_y_test_adv)               
            print(f'{epoch_index+1:04d} epoch at trained classifier accuary on adversarial testset:{epoch_adv_test_accuracy * 100:.4f}%' ) 
            print(f'{epoch_index+1:04d} epoch at trained classifier loss on adversarial testset:{epoch_adv_test_loss}' )    

            #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-adv')
            os.makedirs(tensorboard_log_adv_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_adv_acc_dir)   
            writer_adv_acc = SummaryWriter(log_dir = tensorboard_log_adv_acc_dir, comment= '-'+'advtestacc') 
            writer_adv_acc.add_scalar(tag = "epoch_adv_acc", scalar_value = epoch_adv_test_accuracy, global_step = epoch_index + 1 )
            writer_adv_acc.close()
            #--------------------------------------------------
           #-------------tensorboard实时画图-------------------
            tensorboard_log_adv_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-adv')
            os.makedirs(tensorboard_log_adv_loss_dir,exist_ok=True)    
            writer_adv_loss = SummaryWriter(log_dir = tensorboard_log_adv_loss_dir, comment= '-'+'advtestloss') 
            writer_adv_loss.add_scalar(tag = "epoch_adv_loss", scalar_value = epoch_adv_test_loss, global_step = epoch_index + 1 )
            writer_adv_loss.close()
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-acc-cle')
            os.makedirs(tensorboard_log_cle_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_cle_acc_dir)   
            writer_cle_acc = SummaryWriter(log_dir = tensorboard_log_cle_acc_dir, comment= '-'+'cletestacc') 
            writer_cle_acc.add_scalar(tag = "epoch_cle_acc", scalar_value = epoch_cle_test_accuracy, global_step = epoch_index + 1 )
            writer_cle_acc.close()
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_cle_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-cle')
            os.makedirs(tensorboard_log_cle_loss_dir,exist_ok=True)    
            writer_cle_loss = SummaryWriter(log_dir = tensorboard_log_cle_loss_dir, comment= '-'+'cletestloss') 
            writer_cle_loss.add_scalar(tag = "epoch_cle_loss", scalar_value = epoch_cle_test_loss, global_step = epoch_index + 1 )
            writer_cle_loss.close()
            #--------------------------------------------------

            #-------------tensorboard实时画图-------------------
            tensorboard_log_train_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss-train')
            os.makedirs(tensorboard_log_train_loss_dir,exist_ok=True)    
            writer_tra_loss = SummaryWriter(log_dir = tensorboard_log_train_loss_dir, comment= '-'+'augtrainloss') 
            writer_tra_loss.add_scalar(tag = "epoch_augtrain_loss", scalar_value = epoch_total_loss/len(self._train_dataloader), global_step = epoch_index + 1 )
            writer_tra_loss.close()
            #--------------------------------------------------
          
