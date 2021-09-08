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
        torchvisionmodel_dict = ['resnet34','resnet50','vgg19','densenet169','alexnet','inception_v3']
        if model_name in torchvisionmodel_dict:
            model = self.__gettorchvisionmodel__()
        else:
            model = self.__getlocalmodel__()
        return model

    def __gettorchvisionmodel__(self) ->"torchvision.models":
        model_name = self._args.cla_model
        classes_number = self._args.n_classes
        pretrain_flag = self._args.pretrained_on_imagenet
        torchvisionmodel =  torchvision.models.__dict__[model_name](pretrained=pretrain_flag)
        
        last_name = list(torchvisionmodel._modules.keys())[-1]
        last_module = torchvisionmodel._modules[last_name]
        print('last_name:',last_name)
        print('last_module:',last_module)
        
        if isinstance(last_module, torch.nn.Linear):
            n_features = last_module.in_features
            torchvisionmodel._modules[last_name] = torch.nn.Linear(n_features, classes_number)
        elif isinstance(last_module, torch.nn.Sequential):
            seq_last_name = list(torchvisionmodel._modules.keys())[-1]
            seq_last_module = torchvisionmodel._modules[seq_last_name]
            print('seq_last_name:',seq_last_name)
            print('seq_last_module:',seq_last_module)
            n_features = seq_last_module.in_features
            last_module._modules[seq_last_name] = torch.nn.Linear(n_features, classes_number)
        last = list(torchvisionmodel.named_modules())[-1][1]
        print('torchvisionmodel.last:',last)
        return torchvisionmodel

    def __getlocalmodel__(self)->"CustomNet":
        print('此处需自定义模型')
        local_model = CustomNet()
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
                output = self._model(batch_imgs)

                # print("output:",output)                                     #   output: tensor([[-0.2694,  0.1577,  0.4321,  ...,  0.0562,  0.3836,  0.8319],
                # print("output.shape:",output.shape)                         #   output.shape: torch.Size([256, 10])
                # softmax_output = torch.nn.functional.softmax(output, dim = 1)
                # print("softmax_output:",softmax_output)                     #   softmax_output: tensor([[0.0128, 0.1096, 0.0726,  ..., 0.1264, 0.0353, 0.3227],
                # print("softmax_output.shape:",softmax_output.shape)         #   softmax_output.shape: torch.Size([256, 10])              
                # # raise error

                batch_loss = self._lossfunc(output,batch_labs)
                # batch_loss = self._lossfunc(softmax_output,batch_labs)

                batch_loss.backward()
                self._optimizer.step()

                _, predicted_label_index = torch.max(output.data, 1)   
                # _, predicted_label_index = torch.max(softmax_output.data, 1)    

                batch_correct_num = (predicted_label_index == batch_labs).sum().item()     
                epoch_correct_num += batch_correct_num                                     
                epoch_total_loss += batch_loss

                print("[Epoch %d/%d] [Batch %d/%d] [Batch classify loss: %f] " % (epoch_index, self._args.epochs, batch_index, len(self._train_dataloader), batch_loss.item()))

            #--------当前epoch分类模型在当前训练集epoch上的准确率-------------            
            epoch_train_accuarcy = epoch_correct_num / self._trainset_len
            global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list                  
            epoch_train_loss = epoch_total_loss / self._trainbatch_num
            global_train_loss.append(epoch_train_loss)

            #--------当前epoch分类模型在整体测试集上的准确率------------- 
            epoch_test_accuracy, epoch_test_loss = EvaluateAccuracy(self._model, self._lossfunc, self._test_dataloader)
            global_test_acc.append(epoch_test_accuracy)   
            global_test_loss.append(epoch_test_loss)

            # print(f'{epoch_index:04d} epoch classifier accuary on the current epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
            # print(f'{epoch_index:04d} epoch classifier loss on the current epoch training examples:{epoch_train_loss:.4f}' )   
            print(f'{epoch_index:04d} epoch classifier accuary on the entire testing examples:{epoch_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index:04d} epoch classifier loss on the entire testing examples:{epoch_test_loss:.4f}' )  
            
            if epoch_index % 12 == 0 and epoch_index > 0:
                torch.save(self._model,f'{self._exp_result_dir}/standard-trained-classifier-{self._args.cla_model}-on-clean-{self._args.dataset}-epoch-{epoch_index:04d}.pkl')
            
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
    
    def __adjustlearningrate__(self, epoch_index):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""#   每隔10epoch除以一次10
        # lr = self._args.lr * (0.1 ** (epoch_index // 10))                   #   30//30=1 31//30=1 60//30=2 返回整数部分
        # lr = self._args.lr * (0.1 ** (epoch_index // 5))                   #    每隔5epoch除以一次10

        # if epoch_index<5:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5:
        #     lr = self._args.lr * (0.1 ** (epoch_index // 5))    #   0.001
        # elif epoch_index >= 10:
        #     lr = self._args.lr * (0.1 ** (epoch_index // 10))   #   0.001
        
        # if epoch_index < 5:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5 and epoch_index < 8:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 8 and epoch_index < 11:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 11 :
        #     lr = self._args.lr * 0.1                            #   0.001

        # if epoch_index < 5:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5 and epoch_index < 8:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 8 and epoch_index < 12:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 12 :
        #     lr = self._args.lr * 0.1                            #   0.001

        # if epoch_index < 5:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5 and epoch_index < 10:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 10 and epoch_index < 15:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 15 :
        #     lr = self._args.lr * 0.1                            #   0.001

        # if epoch_index <= 4:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5 and epoch_index <= 7:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 8 and epoch_index <= 10:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 11 :
        #     lr = self._args.lr * 0.1                            #   0.001

        # if epoch_index <= 4:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5 and epoch_index <= 7:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 8 and epoch_index <= 11:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 12 :
        #     lr = self._args.lr * 0.1                            #   0.001

        # if epoch_index <= 4:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5 and epoch_index <= 6:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 7 and epoch_index <= 8:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 9 and epoch_index <= 10:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 11 and epoch_index <= 12:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 13 :
        #     lr = self._args.lr * 0.1                            #   0.001

        # if epoch_index <= 5:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 6 and epoch_index <= 8:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 9 and epoch_index <= 11:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 12 :
        #     lr = self._args.lr * 0.1                            #   0.001

        # if epoch_index <= 4:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 5 and epoch_index <= 7:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 8 and epoch_index <= 10:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 11 and epoch_index <= 13:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 13:
        #     lr = self._args.lr * 0.01                            #   0.0001

        # if epoch_index <= 9:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 10 and epoch_index <= 12:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 13:
        #     lr = self._args.lr * 0.01                            #   0.0001

        # if epoch_index <= 7:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 8 and epoch_index <= 10:
        #     lr = self._args.lr * 0.1                            #   0.001
        # elif epoch_index >= 11:
        #     lr = self._args.lr * 0.01                            #   0.0001

        # if epoch_index <= 9:
        #     lr = self._args.lr                                  #   0.01
        # elif epoch_index >= 10:
        #     lr = self._args.lr * 0.1                            #   0.001

        if epoch_index <= 7:
            lr = self._args.lr                                  #   0.01
        elif epoch_index >= 8:
            lr = self._args.lr * 0.1                            #   0.001

        # lr2 = smooth_step(10,20,30,40,epoch_index)
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        print(f'{epoch_index}epoch learning rate:{lr}')             #   0epoch learning rate:0.01

    def evaluatefromdataloader(self,model,test_dataloader) -> None:
        if torch.cuda.is_available():
            self._lossfunc.cuda()
            # self._model.cuda()    #             check
            model.cuda()
        test_accuracy, test_loss = EvaluateAccuracy(model, self._lossfunc, test_dataloader)     
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

        self._lossfunc = torch.nn.CrossEntropyLoss()

        x_set = x_set.cuda()
        y_set = y_set.cuda()

        # print("x_set.type: ",type(x_set))                   #   x_set.type:  <class 'torch.Tensor'>
        # print("x_set.shape: ",x_set.shape)                  #   x_set.shape:  torch.Size([10000, 3, 32, 32])
        # print("y_set.type: ",type(y_set))                   #   y_set.type:  <class 'torch.Tensor'>
        # print("y_set.shape: ",y_set.shape)                  #   y_set.shape:  torch.Size([10000])

        testset_total_num = len(y_set)
        print("test set total_num:",testset_total_num)
        
        correct_num = 0
        total_loss = 0

        with torch.no_grad():
            # print("flag A x_set[0][0]:",x_set[0][0])                             #   [ 0.6338,  0.6531,  0.7694,  ...,  0.2267,  0.0134, -0.1804]...
            # print("flag A y_set[0]:",y_set[0])                                   #   [3, 8, 8,  ..., 5, 1, 7]tensor

            output = classifier(x_set)
            # print("output:",output)
            loss = self._lossfunc(output,y_set)
            _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index
            correct_num = (predicted_label_index == y_set).sum().item()
            total_loss += loss
            
            # print("测试样本总数：",testset_total_num)
            # print("预测正确总数：",correct_num)
            # print("预测总损失：",total_loss)

            test_accuracy = correct_num / testset_total_num
            test_loss = total_loss / testset_total_num                
        
        # del x_set_tensor
        
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
        aug_x_train = torch.cat([cle_x_train, x_train_mix], dim=0)
        aug_y_train = torch.cat([cle_y_train_onehot, y_train_mix], dim=0)

        # #   扩增混合样本：100mix
        # aug_x_train = x_train_mix
        # aug_y_train = y_train_mix

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
        
        #   混合训练（MMAT） 
        # classify_model.fit(aug_x_train, aug_y_train, x_test_adv, y_test_adv, exp_result_dir, args, nb_epochs=args.epochs, batch_size=args.batch_size)
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
        # print("adv_xset_tensor.shape:",adv_xset_tensor.shape)                                                   #   adv_xset_tensor.shape: torch.Size([10000, 3, 32, 32])
        # print("adv_xset_tensor[0][0]:",adv_xset_tensor[0][0])                                                   #   adv_xset_tensor[0][0]: tensor([[ 0.8338,  0.8531,  0.9694,  ...,  0.0267,  0.0000,  0.0196]
        
        
        # print("adv_yset_tensor.type:",type(adv_yset_tensor))                                                    #   adv_yset_tensor.type: <class 'torch.Tensor'>                                   
        # print("adv_yset_tensor.shape:",adv_yset_tensor.shape)                                                   #   adv_yset_tensor.shape: torch.Size([10000])
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

                softmax_outputs = torch.nn.functional.softmax(output, dim = 1)                   #   对每一行进行softmax
                # print("softmax_outputs.type：",type(softmax_outputs))                         #   
                # print("softmax_outputs.shape",softmax_outputs.shape)                          #   softmax_outputs.shape torch.Size([256, 10])
                # print("softmax_outputs",softmax_outputs)                                      #   softmax_outputs tensor([[4.6068e-05, 5.4200e-05, 2.3084e-04,  ..., 3.3534e-05, 5.0451e-03,
                
                # batch_loss = self._lossfunc(softmax_outputs,batch_labs)
                batch_loss = self.__mixuplossfunc__(softmax_outputs,batch_labs)

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
            global_test_acc.append(epoch_test_accuracy)   
            global_test_loss.append(epoch_test_loss)

            # print(f'{epoch_index:04d} epoch classifier accuary on the current epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
            # print(f'{epoch_index:04d} epoch classifier loss on the current epoch training examples:{epoch_train_loss:.4f}' )   
            print(f'{epoch_index:04d} epoch classifier accuary on the entire testing examples:{epoch_test_accuracy*100:.4f}%' )  
            print(f'{epoch_index:04d} epoch classifier loss on the entire testing examples:{epoch_test_loss:.4f}' )  


        return global_train_acc, global_test_acc, global_train_loss, global_test_loss

    def __evaluatesoftlabelfromtensor__(self, classifier, x_set:Tensor, y_set:Tensor):
    # def evaluatefromtensor(self, classifier, x_set:Tensor, y_set:Tensor):
        if torch.cuda.is_available():
            classifier.cuda()             

        x_set = x_set.cuda()
        y_set = y_set.cuda()

        testset_total_num = len(y_set)
        print("test set total_num:",testset_total_num)
        
        correct_num = 0
        total_loss = 0
        eva_lossfunc = torch.nn.CrossEntropyLoss()
        with torch.no_grad():

            output = classifier(x_set)  
            # print("output:",output)
            # softmax_outputs = torch.nn.functional.softmax(output, dim = 1)                      #   对每一行进行softmax
            # print("softmax_outputs.shape:",softmax_outputs.shape)                               #   output.shape: torch.Size([10000, 10])
            # print("softmax_outputs:",softmax_outputs)                               #   output.shape: torch.Size([10000, 10])
            print("y_set.shape: ",y_set.shape)                  #  y_set.shape:  torch.Size([10000])
            print("y_set: ",y_set)                  #  y_set.shape:  torch.Size([10000])

            loss = eva_lossfunc(output,y_set)
            print("loss.shape:",loss.shape)

            _, predicted_label_index = torch.max(output, 1)        
            print("predicted_label_index.shape:",predicted_label_index.shape)
            
            correct_num = (predicted_label_index == y_set).sum().item()
            total_loss += loss
            
            # print("测试样本总数：",testset_total_num)
            # print("预测正确总数：",correct_num)
            # print("预测总损失：",total_loss)

            test_accuracy = correct_num / testset_total_num
            test_loss = total_loss / testset_total_num                
        
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