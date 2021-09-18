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
# from clamodels.classifier import MaggieClassifier
import copy
import os
from torch import LongTensor

Tensor = torch.Tensor
# class AdvAttack(MaggieClassifier):
#     r"""
#         class of the adversarial attack 
#         attributes:
#         self._args
#         self._model
#         self._loss
#         self._optimizer
#         self._targetmodel
#         self._test_dataloader
#         self._whitebox
#         self._artmodel
#         self._advgenmodel

#         methods:
#         self.__init__()
#         self.__getartmodel__()
#         self.__getadvgenmodel__()
   
#     """
#     def __init__(self, args, learned_model):
        
#         # run the father class MaggieClassifier.__init__()
#         super().__init__(args, learned_model)

#         """
#         此时，
#             self._model = learned_model 
#             self._optimizer = learned_model._optimizer
#             self._lossfunc = learned_model._lossfunc
#         """

#         self._targetmodel = copy.deepcopy(learned_model)    #深拷贝

#         # initilize the attack model used to generate adversarial examples
#         self._whitebox = True                       #   white box attack or black box attack
#         if self._whitebox == True:
#             # self._model = torchvision.models.resnet34(pretrained=True)
#             self._model = learned_model                 #   白盒攻击，攻击与目标模型一样
#         elif self._whitebox == False:
#             self._model = torchvision.models.resnet34(pretrained=True)
#             raise Exception("balck box attack error!")

#         # initilize the art format attack model
#         self._artmodel = self.__getartmodel__()

#         # initilize the generate model of the art format attack model
#         self._advgenmodel = self.__getadvgenmodel__()

#     def targetmodel(self):
#         return self._targetmodel

#     def __getartmodel__(self) -> "PyTorchClassifier":
#         if torch.cuda.is_available():
#             self._lossfunc.cuda()
#             self._model.cuda()      
        
#         data_raw = False                                        #   是否在之前对数据集进行过归一化
#         if data_raw == True:
#             min_pixel_value = 0.0
#             max_pixel_value = 255.0
#         else:
#             min_pixel_value = 0.0
#             max_pixel_value = 1.0        

#         artmodel = PyTorchClassifier(
#             model=self._model,
#             clip_values=(min_pixel_value, max_pixel_value),
#             loss=self._lossfunc,
#             optimizer=self._optimizer,
#             input_shape=(self._args.channels, self._args.img_size, self._args.img_size),
#             nb_classes=self._args.n_classes,
#         )             
#         return artmodel

#     def __getadvgenmodel__(self) -> "art.attacks.evasion":
        
#         if self._args.attack_mode == 'fgsm':                              #   FGSM攻击
#             print('Get FGSM examples generate model')
#             # advgenmodel = art.attacks.evasion.FastGradientMethod(estimator=self._artmodel, eps=0.2, targeted=False)    #   estimator: A trained classifier. eps: Attack step size (input variation).
#             print("self._args.attack_eps:",self._args.attack_eps)
#             advgenmodel = art.attacks.evasion.FastGradientMethod(estimator=self._artmodel, eps=self._args.attack_eps, targeted=False)    #   estimator: A trained classifier. eps: Attack step size (input variation).

#         elif self._args.attack_mode =='deepfool':                         #   DeepFool攻击
#             print('Get DeepFool examples generate model')
#             advgenmodel = art.attacks.evasion.DeepFool(classifier=self._artmodel, epsilon=0.2)                         #   estimator: A trained classifier. eps: Attack step size (input variation).
#         elif self._args.attack_mode =='bim':                              #   BIM攻击
#             print('Get BIM(PGD) examples generate model')
#             advgenmodel = art.attacks.evasion.BasicIterativeMethod(estimator=self._artmodel, eps=0.2, targeted=False)  #   estimator: A trained classifier. eps: Attack step size (input variation).
#         elif self._args.attack_mode =='cw':                               #   CW攻击
#             print('Get CW examples generate model')
#             advgenmodel = art.attacks.evasion.CarliniL2Method(classifier=self._artmodel, targeted=False)               #   estimator: A trained classifier. eps: Attack step size (input variation).
#         elif self._args.attack_mode =='pgd': 
#             advgenmodel = art.attacks.evasion.ProjectedGradientDescent(estimator=self._artmodel, eps=0.2, targeted=False)   #默认eps是0.3
#         elif self._args.attack_mode == None:
#             raise Exception('please input the attack mode')           
#         return advgenmodel

#     def getexpresultdir(self):
#         return self._exp_result_dir
    
#     # def generate(self,train_dataloader,test_dataloader,exp_result_dir) -> "Tensor":
#     def generate(self, exp_result_dir, test_dataloader, train_dataloader = None) -> "Tensor":
#         if train_dataloader is not None:
#             self._train_dataloader = train_dataloader
#             self._test_dataloader = test_dataloader 
#             self._exp_result_dir = exp_result_dir

#             self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
#             os.makedirs(self._exp_result_dir,exist_ok=True)            

#             self._x_train, self._y_train = self.__getsettensor__(self._train_dataloader)
#             self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)

#             """"artmodel.generate()函数生成对抗样本时只接受numpy ndarray输入，所以进行tensor转numpy"""

#             self._x_train = self._x_train.cpu().numpy()                         #   self._x_train原本是GPU
#             self._y_train = self._y_train.cpu().numpy()
#             self._x_test = self._x_test.cpu().numpy()
#             self._y_test = self._y_test.cpu().numpy()

#             print('generating adversarial examples...')
#             self._x_train_adv = self._advgenmodel.generate(x = self._x_train, y = self._y_train)
#             self._y_train_adv = self._y_train
#             self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
#             self._y_test_adv = self._y_test
#             print('finished generate adversarial examples !')

#             #numpy转tensor
#             self._x_train_adv = torch.from_numpy(self._x_train_adv).cuda()
#             self._y_train_adv = torch.from_numpy(self._y_train_adv).cuda()
#             self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
#             self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

#             #   numpy转tensor
#             self._x_train = torch.from_numpy(self._x_train).cuda()
#             self._y_train = torch.from_numpy(self._y_train).cuda()
#             self._x_test = torch.from_numpy(self._x_test).cuda()
#             self._y_test = torch.from_numpy(self._y_test).cuda()

#             self.__saveadvpng__()

#             return self._x_train_adv, self._y_train_adv, self._x_test_adv, self._y_test_adv         #   GPU tensor

#         elif train_dataloader is None:
#             self._test_dataloader = test_dataloader 
#             self._exp_result_dir = exp_result_dir

#             self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
#             os.makedirs(self._exp_result_dir,exist_ok=True)            

#             self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)

#             """"artmodel.generate()函数生成对抗样本时只接受numpy ndarray输入，所以进行tensor转numpy"""

#             self._x_test = self._x_test.cpu().numpy()
#             self._y_test = self._y_test.cpu().numpy()

#             print('generating testset adversarial examples...')
#             self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
#             self._y_test_adv = self._y_test
#             print('finished generate testset adversarial examples !')

#             #numpy转tensor
#             self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
#             self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

#             #   numpy转tensor
#             self._x_test = torch.from_numpy(self._x_test).cuda()
#             self._y_test = torch.from_numpy(self._y_test).cuda()

#             # self.__saveadvpng__()
#             return self._x_test_adv, self._y_test_adv         #   GPU tensor

#     def __saveadvpng__(self):

#         classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#         os.makedirs(f'{self._exp_result_dir}/samples/train/',exist_ok=True)    
#         os.makedirs(f'{self._exp_result_dir}/samples/test/',exist_ok=True)    

#         print(f"Saving {self._args.dataset} trainset  adversarial examples...")
#         for img_index, _ in enumerate(self._x_train_adv):
#             save_adv_img = self._x_train_adv[img_index]
#             save_cle_img = self._x_train[img_index]
#             img_true_label = self._y_train_adv[img_index]
            
#             np.savez(f'{self._exp_result_dir}/samples/train/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_adv_img.cpu().numpy())      #   存投影
       
#         print(f"Saving {self._args.dataset} testset  adversarial examples...")
#         for img_index, _ in enumerate(self._x_test_adv):
#             save_adv_img = self._x_test_adv[img_index]
#             save_cle_img = self._x_test[img_index]
#             img_true_label = self._y_test_adv[img_index]

#             np.savez(f'{self._exp_result_dir}/samples/test/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_adv_img.cpu().numpy())      #   存投影npz, projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])
#             # np.savez(f'{self._exp_result_dir}/samples/test/{img_index:08d}-cle-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_cle_img.cpu().numpy())      #   存投影npz, projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])

#             # load_adv_img = np.load(f'{self._exp_result_dir}/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz')['w']
#             # load_cle_img = np.load(f'{self._exp_result_dir}/{img_index:08d}-cle-{img_true_label}-{classification[int(img_true_label)]}.npz')['w']
            
#             # print("load_adv_img:",load_adv_img)                                                     
#             # print("load_adv_img.dtype:",load_adv_img.dtype)
#             # print("load_adv_img.shape:",load_adv_img.shape)

#             # print("load_cle_img:",load_cle_img)                                                     #   load_cle_img: [[[ 0.63375115  0.6531361   0.7694456  ...  0.22666796  0.01343373   -0.18041542]
#             # print("load_cle_img.dtype:",load_cle_img.dtype)                                         #   load_cle_img.dtype: float32
#             # print("load_cle_img.shape:",load_cle_img.shape)                                         #   load_cle_img.shape: (3, 32, 32)

#             # print("load_adv_img:",torch.tensor(load_adv_img))
#             # print("load_adv_img.dtype:",torch.tensor(load_adv_img).dtype)
#             # print("load_adv_img.shape:",torch.tensor(load_adv_img).shape)

#             # print("load_cle_img:",torch.tensor(load_cle_img))                                       #   load_cle_img: tensor([[[ 0.6338,  0.6531,  0.7694,  ...,  0.2267,  0.0134, -0.1804],
#             # print("load_cle_img.dtype:",torch.tensor(load_cle_img).dtype)                           #   load_cle_img.dtype: torch.float32
#             # print("load_cle_img.shape:",torch.tensor(load_cle_img).shape)                           #   load_cle_img.shape: torch.Size([3, 32, 32])
             
#             # raise Exception("maggie stop")

#             # save_image(save_adv_img, f'{self._exp_result_dir}/samples/test/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.png', nrow=5, normalize=True)
#             # save_image(save_cle_img, f'{self._exp_result_dir}/samples/test/{img_index:08d}-cle-{img_true_label}-{classification[int(img_true_label)]}.png', nrow=5, normalize=True)  

#     def generateadvfromtestsettensor(self, exp_result_dir, testset_tensor_x, testset_tensor_y):
#             self._exp_result_dir = exp_result_dir
#             self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
#             os.makedirs(self._exp_result_dir,exist_ok=True)            

#             # self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)
#             self._x_test = testset_tensor_x
#             self._y_test = testset_tensor_y

#             """"artmodel.generate()函数生成对抗样本时只接受numpy ndarray输入，所以进行tensor转numpy"""
#             self._x_test = self._x_test.cpu().numpy()
#             self._y_test = self._y_test.cpu().numpy()

#             print('generating testset adversarial examples...')
#             self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
#             self._y_test_adv = self._y_test
#             print('finished generate testset adversarial examples !')

#             #numpy转tensor
#             self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
#             self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

#             #   numpy转tensor
#             self._x_test = torch.from_numpy(self._x_test).cuda()
#             self._y_test = torch.from_numpy(self._y_test).cuda()

#             # self.__saveadvpng__()
#             return self._x_test_adv, self._y_test_adv         #   GPU tensor        


class AdvAttack():
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
    def __init__(self, args, learned_model) -> None:                 # 双下划线表示只有Classifier类本身可以访问   ->后是对函数返回值的注释，None表明无返回值
        print('initlize classifier')
        self._args = args
        print("learned calssify model != None")
        self._model = learned_model             #   浅拷贝
        
        self._lossfunc = self.__getlossfunc__()
        self._optimizer = self.__getoptimizer__()
        # self._lossfunc = learned_model._optimizer
        # self._optimizer = learned_model._lossfunc

        self._targetmodel = copy.deepcopy(learned_model)    #深拷贝

        # initilize the attack model used to generate adversarial examples
        self._whitebox = True                       #   white box attack or black box attack
        if self._whitebox == True:
            # self._model = torchvision.models.resnet34(pretrained=True)
            self._model = learned_model                 #   白盒攻击，攻击与目标模型一样
        elif self._whitebox == False:
            self._model = torchvision.models.resnet34(pretrained=True)
            raise Exception("balck box attack error!")

        # initilize the art format attack model
        self._artmodel = self.__getartmodel__()

        # initilize the generate model of the art format attack model
        self._advgenmodel = self.__getadvgenmodel__()
        
    
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

    # def __init__(self, args, learned_model):
        
    #     # run the father class MaggieClassifier.__init__()
    #     super().__init__(args, learned_model)

    #     """
    #     此时，
    #         self._model = learned_model 
    #         self._optimizer = learned_model._optimizer
    #         self._lossfunc = learned_model._lossfunc
    #     """

    #     self._targetmodel = copy.deepcopy(learned_model)    #深拷贝

    #     # initilize the attack model used to generate adversarial examples
    #     self._whitebox = True                       #   white box attack or black box attack
    #     if self._whitebox == True:
    #         # self._model = torchvision.models.resnet34(pretrained=True)
    #         self._model = learned_model                 #   白盒攻击，攻击与目标模型一样
    #     elif self._whitebox == False:
    #         self._model = torchvision.models.resnet34(pretrained=True)
    #         raise Exception("balck box attack error!")

    #     # initilize the art format attack model
    #     self._artmodel = self.__getartmodel__()

    #     # initilize the generate model of the art format attack model
    #     self._advgenmodel = self.__getadvgenmodel__()

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
            advgenmodel = art.attacks.evasion.DeepFool(classifier=self._artmodel, epsilon=0.3)                         #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self._args.attack_mode =='bim':                              #   BIM攻击
            print('Get BIM(PGD) examples generate model')
            advgenmodel = art.attacks.evasion.BasicIterativeMethod(estimator=self._artmodel, eps=0.3, targeted=False)  #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self._args.attack_mode =='cw':                               #   CW攻击
            print('Get CW examples generate model')
            advgenmodel = art.attacks.evasion.CarliniL2Method(classifier=self._artmodel, targeted=False)               #   estimator: A trained classifier. eps: Attack step size (input variation).
        elif self._args.attack_mode =='pgd': 
            advgenmodel = art.attacks.evasion.ProjectedGradientDescent(estimator=self._artmodel, eps=0.3, targeted=False)   #默认eps是0.3
        elif self._args.attack_mode == None:
            raise Exception('please input the attack mode')           
        return advgenmodel

    def getexpresultdir(self):
        return self._exp_result_dir
    
    # def generate(self,train_dataloader,test_dataloader,exp_result_dir) -> "Tensor":
    def generate(self, exp_result_dir, test_dataloader, train_dataloader = None) -> "Tensor":
        if train_dataloader is not None:
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

        elif train_dataloader is None:
            self._test_dataloader = test_dataloader 
            self._exp_result_dir = exp_result_dir

            self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
            os.makedirs(self._exp_result_dir,exist_ok=True)            

            self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)

            """"artmodel.generate()函数生成对抗样本时只接受numpy ndarray输入，所以进行tensor转numpy"""

            self._x_test = self._x_test.cpu().numpy()
            self._y_test = self._y_test.cpu().numpy()

            print('generating testset adversarial examples...')
            self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
            self._y_test_adv = self._y_test
            print('finished generate testset adversarial examples !')

            #numpy转tensor
            self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
            self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

            #   numpy转tensor
            self._x_test = torch.from_numpy(self._x_test).cuda()
            self._y_test = torch.from_numpy(self._y_test).cuda()

            # self.__saveadvpng__()
            return self._x_test_adv, self._y_test_adv         #   GPU tensor

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

    def generateadvfromtestsettensor(self, exp_result_dir, testset_tensor_x, testset_tensor_y):
            self._exp_result_dir = exp_result_dir
            self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
            os.makedirs(self._exp_result_dir,exist_ok=True)            

            # self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)
            self._x_test = testset_tensor_x
            self._y_test = testset_tensor_y

            """"artmodel.generate()函数生成对抗样本时只接受numpy ndarray输入，所以进行tensor转numpy"""
            self._x_test = self._x_test.cpu().numpy()
            self._y_test = self._y_test.cpu().numpy()

            print('generating testset adversarial examples...')
            self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
            self._y_test_adv = self._y_test
            print('finished generate testset adversarial examples !')

            #numpy转tensor
            self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
            self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

            #   numpy转tensor
            self._x_test = torch.from_numpy(self._x_test).cuda()
            self._y_test = torch.from_numpy(self._y_test).cuda()

            # self.__saveadvpng__()
            return self._x_test_adv, self._y_test_adv         #   GPU tensor        


    def evaluatefromtensor(self, classifier, x_set:Tensor, y_set:Tensor):
        if torch.cuda.is_available():
            classifier.cuda()             
        
        batch_size = self._args.batch_size
        testset_total_num = len(x_set)
        batch_num = int( np.ceil( int(testset_total_num) / float(batch_size) ) )
        cla_model_name=self._args.cla_model

        # print("x_set.shape:",x_set.shape)           #   x_set.shape: torch.Size([26032, 3, 32, 32])
        # print("y_set.shape:",y_set.shape)           #   y_set.shape: torch.Size([26032])
        # print("testset_total_num:",testset_total_num)       #   testset_total_num: 26032
        # print("batch_num:",batch_num)                   #   batch_num: 813.5
        # print("batch_size:",batch_size)       #  
        # print("cla_model_name:",cla_model_name)       #   cla_model_name: alexnet

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
                    if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'svhn' or self._args.dataset == 'kmnist' or self._args.dataset == 'cifar10':  #   只有imagenet和svhn kmnist搭配googlenet时是返回一个值
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


        return yset_tensor.cuda()       #   yset_tensor 原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接
