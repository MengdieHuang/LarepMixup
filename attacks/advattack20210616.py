"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""
import torch
from art.estimators.classification import PyTorchClassifier
import torchvision
import datas.dataload  
import numpy as np
from art.attacks.evasion import FastGradientMethod,DeepFool,BasicIterativeMethod,CarliniL2Method
from utils.savepng import save_image
from utils.savetxt import SaveAccuracyTxt
from utils.savetxt import SaveLossTxt
from evaluations.accuracy import EvaluateAccuracy
import datas.dataload
from datas.augdataset import AugCIFAR10
from datas.repdataset import AdvCIFAR10
from torchvision.transforms import transforms


class AdvAttackClassifier:
    def __init__(self,opt,classify_model):
        # initialize the parameters
        self.opt = opt   

        # initilize the target model architecture
        self.target_classify_model = classify_model  

        # initilize the attack model architecture
        self.white_box = True                       #   white box attack or black box attack
        if self.white_box == True:
            self.attack_classify_model = classify_model
        elif self.white_box == False:
            self.attack_classify_model = torchvision.models.resnet34(pretrained=True)

        # initilize the loss function
        self.classify_loss = torch.nn.CrossEntropyLoss()
        
        # initilize the optimizer
        self.classify_optimizer =  torch.optim.Adam(self.attack_classify_model.parameters(), lr=self.opt.lr)
    
        #initilize the art package of attack model (to use art attack algorithms)
        self.art_attack_classify_model = self.GetArtClassifier()

        # get the trained art package of attack model
        self.art_estimator_model = self.GetAdvtAttackClassifier()

    def GetArtClassifier(self):

        if torch.cuda.is_available():
            self.classify_loss.cuda()
            self.attack_classify_model.cuda()      
        
        data_raw = False                                        #   是否在之前对数据集进行过归一化
        if data_raw == True:
            min_pixel_value = 0.0
            max_pixel_value = 255.0
        else:
            min_pixel_value = 0.0
            max_pixel_value = 1.0        

        art_attack_classify_model = PyTorchClassifier(
            model=self.attack_classify_model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=self.classify_loss,
            optimizer=self.classify_optimizer,
            input_shape=(self.opt.channels, self.opt.img_size, self.opt.img_size),
            nb_classes=self.opt.n_classes,
        )             
        return art_attack_classify_model

    def GetAdvtAttackClassifier(self):
        if self.opt.attack_mode == 'fgsm':                              #   FGSM攻击
            print('generating FGSM examples')
            return FastGradientMethod(estimator=self.art_attack_classify_model, eps=0.2, targeted=False)    #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self.opt.attack_mode =='deepfool':                         #   DeepFool攻击
            return DeepFool(classifier=self.art_attack_classify_model, epsilon=0.2)                         #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self.opt.attack_mode =='bim':                              #   BIM攻击
            return BasicIterativeMethod(estimator=self.art_attack_classify_model, eps=0.2, targeted=False)  #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self.opt.attack_mode =='cw':                               #   CW攻击
            return CarliniL2Method(classifier=self.art_attack_classify_model, targeted=False)               #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self.opt.attack_mode == None:
            raise Exception('please input the attack mode')   

    def generate(self,exp_result_dir):
        # initilize the dataloader
        self.train_dataloader, self.test_dataloader = self.data()

        # initilize the data ndArray
        self.x_train,self.y_train,self.x_test,self.y_test = self.GetCleanDataArray()  

        print('generating adversarial examples')
        self.x_train_adv = self.art_estimator_model.generate(x = self.x_train, y = self.y_train)
        self.y_train_adv = self.y_train
        self.x_test_adv = self.art_estimator_model.generate(x = self.x_test, y = self.y_test)
        self.y_test_adv = self.y_test
        print('finished generate adversarial examples')
        # print('x_test_adv:',self.x_test_adv[:3])
        # print('x_test:',self.x_test[:3])

        if torch.cuda.is_available():
            Tensor = torch.cuda.FloatTensor 
        else:
            Tensor = torch.FloatTensor

        # save png
        for img_index, img in enumerate(self.x_test_adv):
            if img_index % 1000 == 0:
                save_adv_img = self.x_test_adv[img_index:img_index+25]
                save_adv_img = Tensor(save_adv_img)
                save_cle_img = self.x_test[img_index:img_index+25]
                save_cle_img = Tensor(save_cle_img)
                print('save_adv_img.shape:',save_adv_img.shape)
                if save_adv_img.size(0) == 25 :
                    save_image(save_adv_img, f'{exp_result_dir}/cle-testset-{img_index}.png', nrow=5, normalize=True)
                    save_image(save_cle_img, f'{exp_result_dir}/adv-testset-{img_index}.png', nrow=5, normalize=True)       

        self.train_set = AugCIFAR10(                                             #   用 torchvision.datasets.MNIST类的构造函数返回值给DataLoader的参数 dataset: torch.utils.data.dataset.Dataset[T_co]赋值 https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
            "/home/data/maggie/cifar10",
            train=True,                                             #   从training.pt创建数据集
            download=False,                                          #   自动从网上下载数据集
            transform=transforms.Compose(
                [
                    transforms.Resize(self.opt.img_size), 
                    transforms.CenterCrop(self.opt.img_size),
                    transforms.ToTensor(), 
                    # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            ),
        )

        self.test_set = AdvCIFAR10(                                             #   用 torchvision.datasets.MNIST类的构造函数返回值给DataLoader的参数 dataset: torch.utils.data.dataset.Dataset[T_co]赋值 https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
            "/home/data/maggie/cifar10",
            train=False,                                             #   从training.pt创建数据集
            download=False,                                          #   自动从网上下载数据集
            transform=transforms.Compose(
                [
                    transforms.Resize(self.opt.img_size), 
                    transforms.CenterCrop(self.opt.img_size),
                    transforms.ToTensor(), 
                    # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            ),
        )

        # self.train_set = self.train_dataloader.dataset
        # self.test_set = self.test_dataloader.dataset

        self.train_set.AugmentData(self.x_train_adv,self.y_train_adv)
        self.test_set.AdvReplaceData(self.x_test_adv,self.y_test_adv)
        self.aug_train_dataloader = datasets.dataload.LoadCIFAR10Train(self.opt,self.train_set)
        self.adv_test_dataloader = datasets.dataload.LoadCIFAR10Test(self.opt,self.test_set)

        return self.aug_train_dataloader, self.adv_test_dataloader


    def data(self):
        if self.opt.dataset == 'mnist':
            return datasets.dataload.LoadMNIST(self.opt)
        elif self.opt.dataset == 'kmnist':
            return datasets.dataload.LoadKMNIST(self.opt)
        elif self.opt.dataset == 'cifar10':
            return datasets.dataload.LoadCIFAR10(self.opt)
        elif self.opt.dataset == 'cifar100':
            return datasets.dataload.LoadCIFAR100(self.opt)
        elif self.opt.dataset == 'imagenet':
            return datasets.dataload.LoadIMAGENET(self.opt)
        elif self.opt.dataset == 'lsun':
            return datasets.dataload.LoadLSUN(self.opt)
        elif self.opt.dataset == 'stl10':
            return datasets.dataload.LoadSTL10(self.opt)

    def GetCleanDataArray(self):
        train_dataloader = self.train_dataloader
        test_dataloader = self.test_dataloader

        if self.opt.dataset == 'cifar10':
            jieduan_num = 1000
            # print("train_dataloader.dataset.__dict__.keys:",train_dataloader.dataset.__dict__.keys())
            total_num = len(train_dataloader.dataset)
            x_train = train_dataloader.dataset.data                                             #   shape是（50000，32，32，3）
            y_train = train_dataloader.dataset.targets

            # x_train = x_train[:jieduan_num]
            # y_train = y_train[:jieduan_num]

            x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)    
            print('trainset total_num :',total_num)
            print('x_train.shape:',x_train.shape)


            total_num = len(test_dataloader.dataset)
            x_test = test_dataloader.dataset.data                                             #   shape是（50000，32，32，3）
            y_test = test_dataloader.dataset.targets

            # x_test = x_test[:jieduan_num]
            # y_test = y_test[:jieduan_num]

            x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)    
            print('testset total_num :',total_num)
            print('x_test.shape:',x_test.shape)    

        elif self.opt.dataset == 'imagenet':
            jieduan_num = 100
            x_train = []                                             #   shape是（50000，32，32，3）
            y_train = []       
            train_set_len = len(train_dataloader.dataset)
            print('trainset len:',train_set_len)

            for index in range(jieduan_num):
                img, label = train_dataloader.dataset.__getitem__(index)
                # print("img.shape:",img.shape)       #   [3,1024,1024]
                x_train.append(img)
                y_train.append(label)      
            x_train = torch.stack(x_train) 
            print("x_train shape:",x_train.shape)    
            x_train = x_train.numpy()
            print("x_train shape:",x_train.shape)    

            x_test = []
            y_test = []
            test_set_len = len(test_dataloader.dataset)
            print('testlen len:',test_set_len)
            for index in range(test_set_len):
                img, label = test_dataloader.dataset.__getitem__(index)
                x_test.append(img)                                          #   张量list
                y_test.append(label)

            x_test = torch.stack(x_test) 
            print("x_test shape:",x_test.shape)    
            x_test = x_test.numpy()
            print("x_test shape:",x_test.shape)    
        
        return x_train,y_train,x_test,y_test


    def evaluate(self,classify_model,test_dataloader,exp_result_dir):
        print('evaluate the trained model on adversarial examples')
        classify_loss= self.classify_loss

        if torch.cuda.is_available():
            classify_model.cuda()
            classify_loss.cuda()

        test_accuracy, test_loss = EvaluateAccuracy(classify_model,classify_loss,test_dataloader)

        return test_accuracy,test_loss
        


