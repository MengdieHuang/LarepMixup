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

from tensorboardX import SummaryWriter

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

        if model_name in torchvisionmodel_dict:
            model = self.__gettorchvisionmodel__()      #   加载torchvision库model
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

            #--------当前epoch分类模型在当前训练集epoch上的准确率-------------            
            epoch_train_accuarcy = epoch_correct_num / self._trainset_len
            global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list                  
            epoch_train_loss = epoch_total_loss / self._trainbatch_num
            global_train_loss.append(epoch_train_loss)

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
            tensorboard_log_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run')
            os.makedirs(tensorboard_log_acc_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_acc = SummaryWriter(log_dir = tensorboard_log_acc_dir, comment= '-'+'testacc')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_acc.add_scalar(tag = "epoch_acc", scalar_value = epoch_test_accuracy, global_step = epoch_index + 1 )
            writer_acc.close()
            # raise error
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss')
            os.makedirs(tensorboard_log_loss_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_loss = SummaryWriter(log_dir = tensorboard_log_loss_dir, comment= '-'+'testloss')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_loss.add_scalar(tag = "epoch_loss", scalar_value = epoch_test_loss, global_step = epoch_index + 1 )
            writer_loss.close()
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
        if torch.cuda.is_available():
            classifier.cuda()             
        
        batch_size = self._args.batch_size
        testset_total_num = len(x_set)
        batch_num = int( np.ceil( int(testset_total_num) / float(batch_size) ) )
        cla_model_name=self._args.cla_model

        # print("x_set.shape:",x_set.shape)           #   x_set.shape: torch.Size([26032, 3, 32, 32])
        print("y_set.shape:",y_set.shape)           #   y_set.shape: torch.Size([26032])
        print("testset_total_num:",testset_total_num)       #   testset_total_num: 26032
        print("batch_num:",batch_num)                   #   batch_num: 813.5
        print("batch_size:",batch_size)       #  
        print("cla_model_name:",cla_model_name)       #   cla_model_name: alexnet

        eva_loss = torch.nn.CrossEntropyLoss()
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
                                
                loss = eva_loss(output,labs)
                _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
                
                batch_same_num = (predicted_label_index == labs).sum().item()
                epoch_correct_num += batch_same_num
                epoch_total_loss += loss


        test_accuracy = epoch_correct_num / testset_total_num
        test_loss = epoch_total_loss / batch_num                  

        return test_accuracy, test_loss

    def settensor(self,dataloader)->"Tensor":
        # self._train_dataloader = train_dataloader
        xset_tensor, yset_tensor = self.__getsettensor__(dataloader)
        return xset_tensor, yset_tensor
    
    def __getsettensor__(self,dataloader)->"Tensor":

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
            
            # print("xset_tensor.type:", type(xset_tensor))                           #   xset_tensor.type: <class 'torch.Tensor'>                                                             
            # print("xset_tensor.shape:",xset_tensor.shape)                           #   xset_tensor.shape: torch.Size([50000, 3, 32, 32])
            # print("xset_tensor[0]: ",xset_tensor[0])                                #   xset_tensor[0]:  tensor([[[-1.2854e+00, -1.5955e+00, -1.4598e+00,  ...,  6.3375e-01                                                                
            # print("xset_tensor[0].shape: ",xset_tensor[0].shape)                    #   xset_tensor[0].shape:  torch.Size([3, 32, 32])

            # img = xset_tensor[0].unsqueeze(0)
            # print("img.shape: ",img.shape)                                          #   img.shape:  torch.Size([1, 3, 32, 32])

            # img = (img + 1) * (255/2)                                                                                 
            # img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0]          #   img.type: <class 'torch.Tensor'>                        
            # print("img.type:", type(img))                                                                   
            # print("img.shape:",img.shape)                                           #   img.shape: torch.Size([32, 32, 3])
            # print("img:",img)                                                       #   img: tensor([[[  0,   0,   0],[  0,   0,   0],


            # raise Exception("maggie stop here")                                    

        elif self._args.dataset == 'cifar100':
            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
                                        

        elif self._args.dataset == 'imagenet':
            jieduan_num = 1000
            # x_ndarray = []                                             #   shape是（50000，32，32，3）
            # dataset_len = len(dataloader.dataset)
            # print('dataset len:',dataset_len)
            # for index in range(jieduan_num):
            # # for index in range(dataset_len):
            #     img, _ = dataloader.dataset.__getitem__(index)
            #     x_ndarray.append(img)
            # x_ndarray = torch.stack(x_ndarray) 
            # print("x_ndarray shape:",x_ndarray.shape)                   #   torch.Size([10000, 3, 256, 256])
            # x_ndarray = x_ndarray.numpy()
            # print("x_ndarray shape:",x_ndarray.shape)                   #   (10000, 3, 256, 256)

            xset_tensor = []
            # for img_index in range(len(dataloader.dataset)):
            for img_index in range(jieduan_num):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
            
            # print("x_ndarray.type:", type(x_ndarray))                           
            # print("x_ndarray.shape:",x_ndarray.shape)                                                                   #   x_ndarray.shape: torch.Size([1000, 3, 256, 256])
            # print("flag A x_ndarray[0][0]: ",x_ndarray[0][0])                                                         
            # print("flag A x_ndarray[0][0].shape: ",x_ndarray[0][0].shape)                                               #   flag A x_ndarray[0][0].shape:  torch.Size([256, 256])                                              

        elif self._args.dataset == 'svhn':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                         
            
            # print("xset_tensor.type:", type(xset_tensor))                                                                   
            # print("xset_tensor.shape:",xset_tensor.shape)           
            # print("flag A xset_tensor[0][0]: ",xset_tensor[0][0])                       # normalized                                                
            # print("flag A xset_tensor[0][0].shape: ",xset_tensor[0][0].shape)           # [3,32,32]               ->[32,32]             

        return xset_tensor.cuda()                                       #   xset_tensor原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接

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


        return yset_tensor.cuda()       #   yset_tensor 原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接

    def adversarialtrain(self,
        args,
        cle_x_train,
        cle_y_train,
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


        # #   扩增对抗样本: 50000 cle_x_train + 50000 x_train_adv
        # aug_x_train = torch.cat([cle_x_train, x_train_adv], dim=0)
        # aug_y_train = torch.cat([cle_y_train, y_train_adv], dim=0)

        #   扩增对抗样本: 50000 x_train_adv
        aug_x_train = x_train_adv
        aug_y_train = y_train_adv

        # print("aug_x_train.type:",type(aug_x_train))            #   aug_x_train.type: <class 'torch.Tensor'>
        # print("aug_x_train.shape:",aug_x_train.shape)           #   aug_x_train.shape: torch.Size([100000, 3, 32, 32])
        # print("aug_y_train.type:",type(aug_y_train))            #   aug_y_train.type: <class 'torch.Tensor'>
        # print("aug_y_train.shape:",aug_y_train.shape)           #   aug_y_train.shape: torch.Size([100000])

        # tensor转numpy
        aug_x_train = aug_x_train.cpu().numpy()
        aug_y_train = aug_y_train.cpu().numpy()
        print("aug_x_train.type:",type(aug_x_train))            #   aug_x_train.type: <class 'numpy.ndarray'>
        print("aug_x_train.shape:",aug_x_train.shape)           #   aug_x_train.shape: (100000, 3, 32, 32)
        print("aug_y_train.type:",type(aug_y_train))            #   aug_y_train.type: <class 'numpy.ndarray'>
        print("aug_y_train.shape:",aug_y_train.shape)           #   aug_y_train.shape: (100000,)


        #   对抗训练    
        classify_model.fit(aug_x_train, aug_y_train, x_test_adv, y_test_adv, exp_result_dir, args, nb_epochs=args.epochs, batch_size=args.batch_size)            

    def mmat(self,
        args,
        cle_x_train,
        cle_y_train,
        x_train_mix,
        y_train_mix,
        x_test_adv, 
        y_test_adv, 
        classify_model: "PyTorchClassifier",
        exp_result_dir
    ):


        # print("x_train_mix.type: ", type(x_train_mix))                                                              #   x_train_mix.type:  <class 'torch.Tensor'>   
        print("x_train_mix.shape: ", x_train_mix.shape)                                                             #   x_train_mix.shape:  torch.Size([29127, 3, 32, 32])
        # print("x_train_mix[0][0]:",x_train_mix[0][0])                                                               #   device='cuda:0') GPU tensor
        
        # print("y_train_mix.type: ", type(y_train_mix))                                                              #   y_train_mix.type:  <class 'torch.Tensor'>
        print("y_train_mix.shape: ", y_train_mix.shape)                                                             #   y_train_mix.shape:  torch.Size([29127, 10])
        # print("y_train_mix[0]:",x_train_mix[0])                                                                     #   device='cuda:0') GPU tensor
        
        #   原始样本标签转one hot
        # print("cle_x_train.type: ", type(cle_x_train))                                                              #   cle_x_train.type:  <class 'torch.Tensor'>
        print("cle_x_train.shape: ", cle_x_train.shape)                                                             #   cle_x_train.shape:  torch.Size([50000, 3, 32, 32])
        # print("cle_x_train[0][0]:",cle_x_train[0][0])                                                               #   device='cuda:0') GPU tensor
        
        # print("cle_y_train.type: ", type(cle_y_train))                                                              #   cle_y_train.type:  <class 'torch.Tensor'>
        # print("cle_y_train.shape: ", cle_y_train.shape)                                                             #   cle_y_train.shape:  torch.Size([5000])
        
        cle_y_train_onehot = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float().cuda()  
        # print("cle_y_train_onehot.shape: ", cle_y_train_onehot.shape)                                               #   cle_y_train_onehot.shape:  torch.Size([50000, 10])        
        # print("cle_y_train_onehot.type: ", type(cle_y_train_onehot))                                                #   cle_y_train_onehot.type:  <class 'torch.Tensor'>
        # print("cle_y_train_onehot.shape: ", cle_y_train_onehot.shape)                                               #   cle_y_train_onehot.shape:  torch.Size([50000, 10])
        # print("cle_y_train_onehot[0]:",cle_y_train_onehot[0])                                                       #   device='cuda:0') GPU tensor

        #   扩增混合样本：aug 50000cle + 100mix
        # cle_x_train.cuda()
        # x_train_mix.cuda()
        # y_train_mix.cuda()
        # aug_x_train = torch.cat([cle_x_train[:7588], x_train_mix], dim=0)
        # aug_y_train = torch.cat([cle_y_train_onehot[:7588], y_train_mix], dim=0)

        # #   扩增混合样本：100mix 只有混合样本
        aug_x_train = x_train_mix
        aug_y_train = y_train_mix

        # print("aug_x_train.type:",type(aug_x_train))                                                                #   aug_x_train.type: <class 'torch.Tensor'>
        # print("aug_x_train.shape:",aug_x_train.shape)                                                               #   aug_x_train.shape: torch.Size([50003, 3, 32, 32])
        # print("aug_x_train[0][0]:",aug_x_train[0][0])                                                               #   device='cuda:0') GPU tensor

        # print("aug_y_train.type:",type(aug_y_train))                                                                #   aug_y_train.type: <class 'torch.Tensor'>
        # print("aug_y_train.shape:",aug_y_train.shape)                                                               #   aug_y_train.shape: torch.Size([50003, 10])
        # print("aug_y_train[0]:",aug_y_train[0])                                                                     #   device='cuda:0') GPU tensor
        
        # tensor转numpy
        aug_x_train = aug_x_train.cpu().numpy()
        aug_y_train = aug_y_train.cpu().numpy()
        print("aug_x_train.type:",type(aug_x_train))                                                                #   aug_x_train.type: <class 'numpy.ndarray'>
        print("aug_x_train.shape:",aug_x_train.shape)                                                               #   aug_x_train.shape: (50003, 3, 32, 32)
        print("aug_y_train.type:",type(aug_y_train))                                                                #   aug_y_train.type: <class 'numpy.ndarray'>
        print("aug_y_train.shape:",aug_y_train.shape)                                                               #   aug_y_train.shape: (50003, 10)
        
        # raise error
        #   混合训练（MMAT） 
        # classify_model.fit(aug_x_train, aug_y_train, x_test_adv, y_test_adv, exp_result_dir, args, nb_epochs=args.epochs, batch_size=args.batch_size)
        # mmat使用以下两种训练代码
        classify_model.fit_softlabel(aug_x_train, aug_y_train, x_test_adv, y_test_adv, exp_result_dir, args, nb_epochs=args.epochs, batch_size=args.batch_size)
        # self.__traintensorset__(aug_x_train,aug_y_train,x_test_adv, y_test_adv,exp_result_dir)

    def getadvset(self,adv_dataset_path):
        adv_xset_tensor, adv_yset_tensor = self.__getadvsettensor__(adv_dataset_path)
        return adv_xset_tensor, adv_yset_tensor     
        
    def __getadvsettensor__(self,adv_dataset_path):

        file_dir=os.listdir(adv_dataset_path)
        file_dir.sort()
        # print("file_dir:",file_dir)                                                                               #   file_dir: ['00000000-adv-3-cat.npz', '00000000-adv-3-cat.png',
        # print("file_dir[0][9:12] :",file_dir[0][9:12])                                                            #   file_dir[0][9:12] : adv
        
        # filenames1 = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz'] 
        # print("filenames1:",filenames1)                                                                           #   filenames1: ['00000000-adv-3-cat.npz', '00000000-cle-3-cat.npz', '00000001-adv-8-ship.npz', '00000001-cle-8-ship.npz',
        filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[9:12] == 'adv']           
        # print("filenames:",filenames)                                                                               #   filenames: ['00000000-adv-3-cat.npz', '00000001-adv-8-ship.npz', '00000002-adv-8-ship.npz'
        # raise Exception("maggie stop here")

        adv_xset_tensor = []
        adv_yset_tensor = []
        for index, filename in enumerate(filenames):
            adv_npz_path = os.path.join(adv_dataset_path,filename)

            load_adv_img = np.load(adv_npz_path)['w']            
            load_adv_img = torch.tensor(load_adv_img)
            
            load_adv_label = int(filename[13:14])
            load_adv_label = torch.tensor(load_adv_label)

            adv_xset_tensor.append(load_adv_img)
            adv_yset_tensor.append(load_adv_label)

        adv_xset_tensor = torch.stack(adv_xset_tensor)                                                                         
        adv_yset_tensor = torch.stack(adv_yset_tensor)   

        # print("adv_xset_tensor.type:",type(adv_xset_tensor))                                                    #   adv_xset_tensor.type: <class 'torch.Tensor'>                                                      
        print("adv_xset_tensor.shape:",adv_xset_tensor.shape)                                                   #   adv_xset_tensor.shape: torch.Size([10000, 3, 32, 32])
        # print("adv_xset_tensor[0][0]:",adv_xset_tensor[0][0])                                                   #   adv_xset_tensor[0][0]: tensor([[ 0.8338,  0.8531,  0.9694,  ...,  0.0267,  0.0000,  0.0196]
        
        
        # print("adv_yset_tensor.type:",type(adv_yset_tensor))                                                    #   adv_yset_tensor.type: <class 'torch.Tensor'>                                   
        print("adv_yset_tensor.shape:",adv_yset_tensor.shape)                                                   #   adv_yset_tensor.shape: torch.Size([10000])
        # print("adv_yset_tensor:",adv_yset_tensor)                                                               #   adv_yset_tensor[0]: tensor(3)

        return adv_xset_tensor, adv_yset_tensor     

    def __traintensorset__(self,train_tensorset_x,train_tensorset_y,test_tensorset_x,test_tensorset_y,exp_result_dir):
            print("输入张量集训练")

            self._train_tensorset_x = torch.tensor(train_tensorset_x)
            self._train_tensorset_y = torch.tensor(train_tensorset_y)
            print("self._train_tensorset_x.shape:",self._train_tensorset_x.shape)                       #  self._train_tensorset.shape: torch.Size([5223, 3, 32, 32])
            print("self._train_tensorset_y.shape:",self._train_tensorset_y.shape)                         #   self._test_tensorset.shape: torch.Size([5223, 10])

            self._test_tensorset_x = torch.tensor(test_tensorset_x)
            self._test_tensorset_y = torch.tensor(test_tensorset_y)
            print("self._test_tensorset_x.shape:",self._test_tensorset_x.shape)                       #  self._train_tensorset.shape: torch.Size([5223, 3, 32, 32])
            print("self._test_tensorset_y.shape:",self._test_tensorset_y.shape)                         #   self._test_tensorset.shape: torch.Size([5223, 10])
                

            self._exp_result_dir = exp_result_dir
            if self._args.defense_mode == "mmat":
                self._exp_result_dir = os.path.join(self._exp_result_dir,f'mmat-{self._args.dataset}-dataset')

            elif self._args.defense_mode == "at":
                self._exp_result_dir = os.path.join(self._exp_result_dir,f'at-{self._args.dataset}-dataset')
            os.makedirs(self._exp_result_dir,exist_ok=True)            


            self._lossfunc = torch.nn.MSELoss() # 支持softlabel计算的函数

            if torch.cuda.is_available():
                self._lossfunc.cuda()
                self._model.cuda()       

            global_train_acc, global_test_acc, global_train_loss, global_test_loss = self.__traintensorsetloop__()
            
            if self._args.defense_mode == "mmat":
                accuracy_png_name = f'manifold mixup adversarial trained classifier {self._args.cla_model} accuracy on adversarial {self._args.dataset}'
                loss_png_name = f'manifold mixup adversarial trained classifier {self._args.cla_model} loss on adversarial {self._args.dataset}'   
        
            elif self._args.defense_mode == "at":
                accuracy_png_name = f'adversarial trained classifier {self._args.cla_model} accuracy on adversarial {self._args.dataset}'
                loss_png_name = f'adversarial trained classifier {self._args.cla_model} loss on adversarial {self._args.dataset}'   
                
            SaveAccuracyCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_acc, global_test_acc, accuracy_png_name)

            SaveLossCurve(self._args.cla_model, self._args.dataset, self._exp_result_dir, global_train_loss, global_test_loss, loss_png_name)

    def __traintensorsetloop__(self):

        print("self._train_tensorset_x.type:",type(self._train_tensorset_x))            #   self._train_tensorset.type: <class 'torch.Tensor'>
        print("self._train_tensorset_x.shape:",self._train_tensorset_x.shape)           #   self._train_tensorset.shape: torch.Size([5223, 3, 32, 32])
        print("self._train_tensorset_x.size:",self._train_tensorset_x.size)             #   self._train_tensorset.size: <built-in method size of Tensor object at 0x7fb4d6833550>
        
        trainset_len = len(self._train_tensorset_x)
        epoch_num = self._args.epochs                                               
        batchsize = self._args.batch_size
        batch_size = batchsize
        batch_num = int(np.ceil(trainset_len / float(batch_size)))
        print("trainset_len:",trainset_len)                                         #   trainset_len: 5223
        print("epoch_num:",epoch_num)                                               #   epoch_num: 12
        print("batch_size:",batch_size)                                             #   batch_size: 256
        print("batch_num:",batch_num)                                               #   batch_num: 21

        shuffle_index = np.arange(trainset_len)
        shuffle_index = torch.tensor(shuffle_index)
        print("shuffle_index:",shuffle_index)                                       #   shuffle_index: tensor([   0,    1,    2,  ..., 5220, 5221, 5222])

        global_train_acc = []
        global_test_acc = []
        global_train_loss = []
        global_test_loss = []

        for epoch_index in range (epoch_num):

            random.shuffle(shuffle_index)
            print("shuffle_index:",shuffle_index)                                       #   shuffle_index: tensor([   0,    1,    2,  ..., 5220, 5221, 5222])

            self.__adjustlearningrate__(epoch_index)     

            epoch_correct_num = 0
            epoch_total_loss = 0

            for batch_index in range (batch_num):

                # x_trainbatch = self._train_tensorset_x[batch_index * batch_size : (batch_index + 1) * batch_size]
                # y_trainbatch = self._train_tensorset_y[batch_index * batch_size : (batch_index + 1) * batch_size]                                                
                
                # print("x_trainbatch.type:",type(x_trainbatch))                          #   x_trainbatch.type: <class 'torch.Tensor'>
                # print("x_trainbatch:",x_trainbatch)                                     #   x_trainbatch: tensor([[[[ 0.9651,  0.9620,  0.9408,  ..., -0.0720, -0.1872, -0.3096]
                # print("x_trainbatch.shape:",x_trainbatch.shape)                         #   x_trainbatch.shape: torch.Size([256, 3, 32, 32])
                # print("y_trainbatch.type:",type(y_trainbatch))                          #   y_trainbatch.type: <class 'torch.Tensor'>
                # print("y_trainbatch:",y_trainbatch)                                     #   y_trainbatch: tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.8523],
                # print("y_trainbatch.shape:",y_trainbatch.shape)                            #   y_trainbatch.shape: torch.Size([256, 10])

                x_trainbatch = self._train_tensorset_x[shuffle_index[batch_index * batch_size : (batch_index + 1) * batch_size]]
                y_trainbatch = self._train_tensorset_y[shuffle_index[batch_index * batch_size : (batch_index + 1) * batch_size]]                                                

                # print("x_trainbatch.type:",type(x_trainbatch))                          #   x_trainbatch.type: <class 'torch.Tensor'>
                # print("x_trainbatch:",x_trainbatch)                                     #   x_trainbatch: tensor([[[[ 0.9651,  0.9620,  0.9408,  ..., -0.0720, -0.1872, -0.3096],
                # print("x_trainbatch.shape:",x_trainbatch.shape)                         #   x_trainbatch.shape: torch.Size([256, 3, 32, 32])
                # print("y_trainbatch.type:",type(y_trainbatch))                          #   
                # print("y_trainbatch:",y_trainbatch)                                     
                # print("y_trainbatch.shape:",y_trainbatch.shape)                             #   y_trainbatch.shape: torch.Size([256, 10])  



                batch_imgs = x_trainbatch.cuda()
                batch_labs = y_trainbatch.cuda()

                self._optimizer.zero_grad()
                output = self._model(batch_imgs)
                # print("output.type：",type(output))                         #   output.type： <class 'torch.Tensor'>
                # print("output.shape",output.shape)                          #   output.shape torch.Size([256, 10])
                # print("output",output)                                      #   output tensor([[ -5.2418,  -4.3187,  -4.6751,  ...,  -5.0795,  -0.7883,   4.9619],

                softmax_outputs = torch.nn.functional.softmax(output, dim = 1)          #   对每一行进行softmax
                
                # print("softmax_outputs.type：",type(softmax_outputs))                   #   
                print("softmax_outputs.shape",softmax_outputs.shape)                    #   softmax_outputs.shape torch.Size([256, 10])
                # print("softmax_outputs",softmax_outputs)                                #   softmax_outputs tensor([[4.6068e-05, 5.4200e-05, 2.3084e-04,  ..., 3.3534e-05, 5.0451e-03,
                # print("self._lossfunc:",self._lossfunc)
                # print("batch_labs.shape:",batch_labs.shape)
                # print("batch_labs:",batch_labs)

                # raise error
                batch_loss = self._lossfunc(softmax_outputs,batch_labs)
                # batch_loss = self.__mixuplossfunc__(softmax_outputs,batch_labs)
                batch_loss.backward()
                self._optimizer.step()

                _, predicted_label_index = torch.max(output.data, 1)    
                # print("predicted_label_index.type:",type(predicted_label_index))            #   predicted_label_index.type: <class 'torch.Tensor'>
                # print("predicted_label_index.shape:",predicted_label_index.shape)           #   predicted_label_index.shape: torch.Size([256]) 给出每一个softlabel中最大的位置index
                # print("predicted_label_index:",predicted_label_index)                       #   predicted_label_index: tensor([9, 9, 9, 4, 4, 9, 9, 7, 3, 9, 4, 3, 3, 3, 4, 4, 1, 1,


                _, batch_labs_maxindex = torch.max(batch_labs, 1)
                # print("batch_labs_maxindex.type:",type(batch_labs_maxindex))                #   batch_labs_maxindex.type: <class 'torch.Tensor'>        
                # print("batch_labs_maxindex.shape:",batch_labs_maxindex.shape)               #   batch_labs_maxindex.shape: torch.Size([256])
                # print("batch_labs_maxindex:",batch_labs_maxindex)                           #   batch_labs_maxindex: tensor([9, 9, 9, 4, 4, 9, 9, 7, 3, 9, 4, 3, 3, 3, 4, 4, 1,

                batch_correct_num = (predicted_label_index == batch_labs_maxindex).sum().item()     

                epoch_correct_num += batch_correct_num                                     
                epoch_total_loss += batch_loss

                print("[Epoch %d/%d] [Batch %d/%d] [Batch classify loss: %f] " % (epoch_index, epoch_num, batch_index, batch_num, batch_loss.item()))
                # raise error
                
            # #--------当前epoch分类模型在当前训练集epoch上的准确率-------------            
            # epoch_train_accuarcy = epoch_correct_num / trainset_len
            # global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list                  
            # epoch_train_loss = epoch_total_loss / batch_num
            # global_train_loss.append(epoch_train_loss)

            # #--------当前epoch分类模型在整体测试集上的准确率------------- 
            # epoch_test_accuracy, epoch_test_loss = EvaluateAccuracy(self._model, self._lossfunc, self._test_dataloader)
            # global_test_acc.append(epoch_test_accuracy)   
            # global_test_loss.append(epoch_test_loss)

            # # print(f'{epoch_index:04d} epoch classifier accuary on the current epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
            # # print(f'{epoch_index:04d} epoch classifier loss on the current epoch training examples:{epoch_train_loss:.4f}' )   
            # print(f'{epoch_index:04d} epoch classifier accuary on the entire testing examples:{epoch_test_accuracy*100:.4f}%' )  
            # print(f'{epoch_index:04d} epoch classifier loss on the entire testing examples:{epoch_test_loss:.4f}' )  


            #--------maggie add---------
            #   当前epoch分类模型在当前训练集epoch上的准确率    
            epoch_train_accuarcy = epoch_correct_num / trainset_len
            global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list                  
            epoch_train_loss = epoch_total_loss / batch_num
            global_train_loss.append(epoch_train_loss)

            #   当前epoch分类模型在整体测试集上的准确率
            epoch_test_accuracy, epoch_test_loss = self.__evaluatesoftlabelfromtensor__(self._model, self._test_tensorset_x, self._test_tensorset_y)

            # epoch_test_accuracy, epoch_test_loss = self.evaluatefromtensor(self._model, self._test_tensorset_x, self._test_tensorset_y)

            
            global_test_acc.append(epoch_test_accuracy)   
            global_test_loss.append(epoch_test_loss)

            # print(f'{epoch_index:04d} epoch classifier accuary on the current epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
            # print(f'{epoch_index:04d} epoch classifier loss on the current epoch training examples:{epoch_train_loss:.4f}' )   
            print(f'{epoch_index:04d} epoch classifier accuary on the entire testing examples:{epoch_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index:04d} epoch classifier loss on the entire testing examples:{epoch_test_loss:.4f}' )  


            # if (epoch_index+1) % 11== 0 and epoch_index > 0:
            if (epoch_index+1)  == 10:
                torch.save(self._model,f'{self._exp_result_dir}/rmt-trained-classifier-{self._args.cla_model}-on-clean-{self._args.dataset}-epoch-{epoch_index+1:04d}.pkl')
            
            #-------------tensorboard实时画图-------------------
            tensorboard_log_acc_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run')
            os.makedirs(tensorboard_log_acc_dir,exist_ok=True)    
            print("tensorboard_log_dir:",tensorboard_log_acc_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_acc = SummaryWriter(log_dir = tensorboard_log_acc_dir, comment= '-'+'testacc')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_acc.add_scalar(tag = "epoch_acc", scalar_value = epoch_test_accuracy, global_step = epoch_index + 1 )
            writer_acc.close()
            # raise error
            #--------------------------------------------------

           #-------------tensorboard实时画图-------------------
            tensorboard_log_loss_dir = os.path.join(self._exp_result_dir,f'tensorboard-log-run-loss')
            os.makedirs(tensorboard_log_loss_dir,exist_ok=True)    
            # print("tensorboard_log_dir:",tensorboard_log_dir)   
            #   tensorboard_log_dir: result/train/cla-train/resnet34-cifar10/20210906/00000/train-cifar10-dataset/tensorboard-log-run

            writer_loss = SummaryWriter(log_dir = tensorboard_log_loss_dir, comment= '-'+'testloss')#  f'{self._args.dataset}-{self._args.cla_model}
            writer_loss.add_scalar(tag = "epoch_loss", scalar_value = epoch_test_loss, global_step = epoch_index + 1 )
            writer_loss.close()
            # raise error
            #--------------------------------------------------



        return global_train_acc, global_test_acc, global_train_loss, global_test_loss

    def __evaluatesoftlabelfromtensor__(self, classifier, x_set:Tensor, y_set:Tensor):
    # def evaluatefromtensor(self, classifier, x_set:Tensor, y_set:Tensor):
        if torch.cuda.is_available():
            classifier.cuda()             

        eva_lossfunc = torch.nn.CrossEntropyLoss()

        
        #---------------maggie------

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
        # raise error

        classify_loss = self._lossfunc
        epoch_correct_num = 0
        epoch_total_loss = 0

        for batch_index in range(batch_num):                                                #   进入batch迭代 共有num_batch个batch
            images = x_set[batch_index * batch_size : (batch_index + 1) * batch_size]
            labels = y_set[batch_index * batch_size : (batch_index + 1) * batch_size]                                                
            # print("images.shape:",images.shape)         # images.shape: torch.Size([32, 3, 32, 32])
            # print("labels.shape:",labels.shape)         #   labels.shape: torch.Size([32])


            # print("cle_x_trainbatch.type:",type(cle_x_trainbatch))                          #   cle_x_trainbatch.type: <class 'numpy.ndarray'>
            # print("cle_x_trainbatch:",cle_x_trainbatch)                                     #   cle_x_trainbatch: [[[[ 59  62  63][ 43  46  45]
            # print("cle_x_trainbatch.shape:",cle_x_trainbatch.shape)                         #   cle_x_trainbatch.shape: (256, 32, 32, 3)
            # print("cle_y_trainbatch.type:",type(cle_y_trainbatch))                          #   cle_y_trainbatch.type: <class 'list'>
            # print("cle_y_trainbatch:",cle_y_trainbatch)                                     #   cle_y_trainbatch: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7,
            # print("cle_y_trainbatch.len:",len(cle_y_trainbatch))                            #   cle_y_trainbatch.len: 256

            imgs = images.cuda()
            labs = labels.cuda()


            with torch.no_grad():

                # output = classifier(imgs)

                if cla_model_name == 'inception_v3':
                    output, aux = classifier(imgs)
                
                elif cla_model_name == 'googlenet':
                    if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'svhn':  #   只有imagenet和svhn搭配googlenet时是返回一个值
                        output = classifier(imgs)
                    else:
                        output, aux1, aux2 = classifier(imgs)
                else:
                    output = classifier(imgs)         
                
                
                # print("output:",output)                                     #   output: tensor([[-3.9918e+00, -4.0301e+00,  6.1573e+00,  ...,  1.5459e+00
                # print("output.shape:",output.shape)                         #   output.shape: torch.Size([256, 10])
                # softmax_output = torch.nn.functional.softmax(output, dim = 1)
                # print("softmax_output:",softmax_output)                     #   softmax_output: tensor([[2.6576e-05, 2.5577e-05, 6.7951e-01,  ..., 6.7526e-03, 4.7566e-05,,
                # print("softmax_output.shape:",softmax_output.shape)         #   softmax_output.shape: torch.Size([256, 10])              
                # raise Exception("maggie error 20210906")

                loss = eva_lossfunc(output,labs)
                _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
                
                # loss = classify_loss(softmax_output,labs)
                # _, predicted_label_index = torch.max(softmax_output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
                            
                
                # print("predicted_label_index:",predicted_label_index)                                                           #   predicted_label_index: tensor([1, 4, 0, 6, 0, 0, 7, 8, 0, 3, 3, 0, 7, 4, 9, 3], device='cuda:0')
                # print("labs:",labs)                                                                                             #   labs: tensor([1, 6, 8, 2, 8, 8, 5, 0, 1, 1, 9, 0, 7, 4, 1, 2], device='cuda:0')
                batch_same_num = (predicted_label_index == labs).sum().item()
                epoch_correct_num += batch_same_num
                epoch_total_loss += loss


        test_accuracy = epoch_correct_num / testset_total_num
        test_loss = epoch_total_loss / batch_num                  
        #---------------------------    
        #--------------old-----------------


        # testset_total_num = len(y_set)
        # print("test set total_num:",testset_total_num)
        
        # correct_num = 0
        # total_loss = 0
        # eva_lossfunc = torch.nn.CrossEntropyLoss()
        # with torch.no_grad():

        #     output = classifier(x_set)  
        #     # print("output:",output)
        #     # softmax_outputs = torch.nn.functional.softmax(output, dim = 1)                      #   对每一行进行softmax
        #     # print("softmax_outputs.shape:",softmax_outputs.shape)                               #   output.shape: torch.Size([10000, 10])
        #     # print("softmax_outputs:",softmax_outputs)                               #   output.shape: torch.Size([10000, 10])
        #     print("y_set.shape: ",y_set.shape)                  #  y_set.shape:  torch.Size([10000])
        #     print("y_set: ",y_set)                  #  y_set.shape:  torch.Size([10000])

        #     loss = eva_lossfunc(output,y_set)
        #     print("loss.shape:",loss.shape)

        #     _, predicted_label_index = torch.max(output, 1)        
        #     print("predicted_label_index.shape:",predicted_label_index.shape)
            
        #     correct_num = (predicted_label_index == y_set).sum().item()
        #     total_loss += loss
            
        #     # print("测试样本总数：",testset_total_num)
        #     # print("预测正确总数：",correct_num)
        #     # print("预测总损失：",total_loss)

        #     test_accuracy = correct_num / testset_total_num
        #     test_loss = total_loss / testset_total_num                
        
        return test_accuracy, test_loss
    #-----------------------------

    def __mixuplossfunc__(self,softmax_outputs,batch_labs):
        criterion =  torch.nn.CrossEntropyLoss()
        print("batch_labs:",batch_labs)                                          #  batch_labs.shape:torch.Size([256, 10])  batch_labs_maxindex.shape: torch.Size([256])
       
        lam, w1_label_index = torch.max(batch_labs, 1)                            #  torch.Size[256]
        print(f'w1_label_index = {int(w1_label_index)}')
        print(f'lam = {lam}')
        
        modified_mixed_label = copy.deepcopy(batch_labs)

        for i in range(self._args.batch_size):
            modified_mixed_label[i][w1_label_index[i]] = 0                             
        print("modified_mixed_label:",modified_mixed_label)                                                                 #   [[0,0,0,...,0]] modified_mixed_label.shape=[1,10]
        
        # 当两个样本是同一类时,将最大置零后，会使得标签2被随机分配为label 0，例如[0,0,0,1,0,0]

        # print("torch.nonzero(modified_mixed_label[0]): ",torch.nonzero(modified_mixed_label[0]))                            #   torch.nonzero([0,0,0,...,0]) = tensor[] 其中size(0,1)
        # print("torch.nonzero(modified_mixed_label[0]).size(0): ",torch.nonzero(modified_mixed_label[0]).size(0))            #   torch.size(0,1)
        

        lam_b, w2_label_index = torch.max(modified_mixed_label, 1)
        print(f'w2_label_index = {int(w2_label_index)}')
        print(f'lam_b = {lam_b}')

        for i in range(self._args.batch_size):
            if torch.nonzero(modified_mixed_label[i]).size(0) == 0:
                w2_label_index[i] = w1_label_index[i]
            print(" test ")
        
        loss_a = criterion(softmax_outputs, w1_label_index)
        loss_b = criterion(softmax_outputs, w2_label_index)
        loss = lam * loss_a + (1 - lam) * loss_b

        return loss

    def __adjustlearningrate__(self, epoch_index):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""#   每隔10epoch除以一次10
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
        # lr2 = smooth_step(10,20,30,40,epoch_index)
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        print(f'{epoch_index}epoch learning rate:{lr}')             #   0epoch learning rate:0.01




