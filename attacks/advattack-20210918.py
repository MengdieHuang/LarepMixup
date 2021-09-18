"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""
import torch
from art.estimators.classification import PyTorchClassifier
from torch.cuda import device
import torchvision
import numpy as np
import art.attacks.evasion
from utils.savepng import save_image
from clamodels.classifier import MaggieClassifier
import copy
import os

class AdvAttack(MaggieClassifier):
    r"""
        class of the adversarial attack 
        attributes:
        self._args
        self._model
        self._loss
        self._optimizer
        self._targetmodel
        self._test_dataloader
        self._whitebox
        self._artmodel
        self._advgenmodel

        methods:
        self.__init__()
        self.__getartmodel__()
        self.__getadvgenmodel__()
   
    """
    def __init__(self, args, learned_model):
        
        # run the father class MaggieClassifier.__init__()
        super().__init__(args, learned_model)

        """
        此时，
            self._model = learned_model 
            self._optimizer = learned_model._optimizer
            self._lossfunc = learned_model._lossfunc
        """

        # initilize the target model to be attacked
        # print("before learned_model:",learned_model)
        self._targetmodel = copy.deepcopy(learned_model)    #深拷贝
        # print("self._targetmodel:",self._targetmodel)
        # print("after learned_model:",learned_model)

        # print('self._targetmodel:',self._targetmodel.fc)
        # print('self._targetmodel:',self._targetmodel)

        # initilize the attack model used to generate adversarial examples
        self._whitebox = True                       #   white box attack or black box attack
        if self._whitebox == True:
            # self._model = torchvision.models.resnet34(pretrained=True)
            self._model = learned_model                 #   白盒攻击，攻击与目标模型一样
        elif self._whitebox == False:
            self._model = torchvision.models.resnet34(pretrained=True)
            raise Exception("balck box attack error!")
        # print('self._attackmodel',self._model.fc)
        # print('self._attackmodel',self._model)

        # initilize the art format attack model
        self._artmodel = self.__getartmodel__()

        # initilize the generate model of the art format attack model
        self._advgenmodel = self.__getadvgenmodel__()

    def targetmodel(self):
        return self._targetmodel

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

    def __getadvgenmodel__(self) -> "art.attacks.evasion":
        
        if self._args.attack_mode == 'fgsm':                              #   FGSM攻击
            print('Get FGSM examples generate model')
            # advgenmodel = art.attacks.evasion.FastGradientMethod(estimator=self._artmodel, eps=0.2, targeted=False)    #   estimator: A trained classifier. eps: Attack step size (input variation).
            print("self._args.attack_eps:",self._args.attack_eps)
            advgenmodel = art.attacks.evasion.FastGradientMethod(estimator=self._artmodel, eps=self._args.attack_eps, targeted=False)    #   estimator: A trained classifier. eps: Attack step size (input variation).

        elif self._args.attack_mode =='deepfool':                         #   DeepFool攻击
            print('Get DeepFool examples generate model')
            advgenmodel = art.attacks.evasion.DeepFool(classifier=self._artmodel, epsilon=0.2)                         #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self._args.attack_mode =='bim':                              #   BIM攻击
            print('Get BIM(PGD) examples generate model')
            advgenmodel = art.attacks.evasion.BasicIterativeMethod(estimator=self._artmodel, eps=0.2, targeted=False)  #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self._args.attack_mode =='cw':                               #   CW攻击
            print('Get CW examples generate model')
            advgenmodel = art.attacks.evasion.CarliniL2Method(classifier=self._artmodel, targeted=False)               #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self._args.attack_mode =='pgd': 
            advgenmodel = art.attacks.evasion.ProjectedGradientDescent(estimator=self._artmodel, eps=0.2, targeted=False)   #默认eps是0.3
        elif self._args.attack_mode == None:
            raise Exception('please input the attack mode')           
        return advgenmodel

    def getexpresultdir(self):
        return self._exp_result_dir
    
    def generate(self,train_dataloader,test_dataloader,exp_result_dir) -> "Tensor":
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader 
        self._exp_result_dir = exp_result_dir

        self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
        os.makedirs(self._exp_result_dir,exist_ok=True)            

        self._x_train, self._y_train = self.__getsettensor__(self._train_dataloader)
        self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)

        """"artmodel.generate()函数生成对抗样本时只接受numpy ndarray输入，所以进行tensor转numpy"""

        self._x_train = self._x_train.cpu().numpy()                         #   self._x_train原本是GPU
        self._y_train = self._y_train.cpu().numpy()
        self._x_test = self._x_test.cpu().numpy()
        self._y_test = self._y_test.cpu().numpy()

        print('generating adversarial examples...')
        self._x_train_adv = self._advgenmodel.generate(x = self._x_train, y = self._y_train)
        self._y_train_adv = self._y_train
        self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
        self._y_test_adv = self._y_test
        print('finished generate adversarial examples !')

        #numpy转tensor
        self._x_train_adv = torch.from_numpy(self._x_train_adv).cuda()
        self._y_train_adv = torch.from_numpy(self._y_train_adv).cuda()
        self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
        self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

        #   numpy转tensor
        self._x_train = torch.from_numpy(self._x_train).cuda()
        self._y_train = torch.from_numpy(self._y_train).cuda()
        self._x_test = torch.from_numpy(self._x_test).cuda()
        self._y_test = torch.from_numpy(self._y_test).cuda()

        self.__saveadvpng__()

        return self._x_train_adv, self._y_train_adv, self._x_test_adv, self._y_test_adv         #   GPU tensor

    def __saveadvpng__(self):
        classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        os.makedirs(f'{self._exp_result_dir}/samples/train/',exist_ok=True)    
        os.makedirs(f'{self._exp_result_dir}/samples/test/',exist_ok=True)    

        print(f"Saving {self._args.dataset} trainset  adversarial examples...")
        for img_index, _ in enumerate(self._x_train_adv):
            save_adv_img = self._x_train_adv[img_index]
            save_cle_img = self._x_train[img_index]
            img_true_label = self._y_train_adv[img_index]
            
            np.savez(f'{self._exp_result_dir}/samples/train/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_adv_img.cpu().numpy())      #   存投影
       
        print(f"Saving {self._args.dataset} testset  adversarial examples...")
        for img_index, _ in enumerate(self._x_test_adv):
            save_adv_img = self._x_test_adv[img_index]
            save_cle_img = self._x_test[img_index]
            img_true_label = self._y_test_adv[img_index]

            np.savez(f'{self._exp_result_dir}/samples/test/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_adv_img.cpu().numpy())      #   存投影npz, projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])
            # np.savez(f'{self._exp_result_dir}/samples/test/{img_index:08d}-cle-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_cle_img.cpu().numpy())      #   存投影npz, projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])

            # load_adv_img = np.load(f'{self._exp_result_dir}/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz')['w']
            # load_cle_img = np.load(f'{self._exp_result_dir}/{img_index:08d}-cle-{img_true_label}-{classification[int(img_true_label)]}.npz')['w']
            
            # print("load_adv_img:",load_adv_img)                                                     
            # print("load_adv_img.dtype:",load_adv_img.dtype)
            # print("load_adv_img.shape:",load_adv_img.shape)

            # print("load_cle_img:",load_cle_img)                                                     #   load_cle_img: [[[ 0.63375115  0.6531361   0.7694456  ...  0.22666796  0.01343373   -0.18041542]
            # print("load_cle_img.dtype:",load_cle_img.dtype)                                         #   load_cle_img.dtype: float32
            # print("load_cle_img.shape:",load_cle_img.shape)                                         #   load_cle_img.shape: (3, 32, 32)

            # print("load_adv_img:",torch.tensor(load_adv_img))
            # print("load_adv_img.dtype:",torch.tensor(load_adv_img).dtype)
            # print("load_adv_img.shape:",torch.tensor(load_adv_img).shape)

            # print("load_cle_img:",torch.tensor(load_cle_img))                                       #   load_cle_img: tensor([[[ 0.6338,  0.6531,  0.7694,  ...,  0.2267,  0.0134, -0.1804],
            # print("load_cle_img.dtype:",torch.tensor(load_cle_img).dtype)                           #   load_cle_img.dtype: torch.float32
            # print("load_cle_img.shape:",torch.tensor(load_cle_img).shape)                           #   load_cle_img.shape: torch.Size([3, 32, 32])
             
            # raise Exception("maggie stop")

            # save_image(save_adv_img, f'{self._exp_result_dir}/samples/test/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.png', nrow=5, normalize=True)
            # save_image(save_cle_img, f'{self._exp_result_dir}/samples/test/{img_index:08d}-cle-{img_true_label}-{classification[int(img_true_label)]}.png', nrow=5, normalize=True)  
