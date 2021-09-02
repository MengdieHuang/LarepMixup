"""
Author: maggie
Date:   2021-06-22
Place:  Xidian University
@copyright
"""

from logging import error, exception
from random import sample
from numpy.core.fromnumeric import shape
import torch
from torch.nn.functional import interpolate
import genmodels.gan
import genmodels.acgan
import genmodels.aae
import genmodels.vae
import genmodels.stylegan2
import genmodels.stylegan2ada
import numpy as np
import os
class CustomGenNet(torch.nn.Module):                                                                                            #   此处是自定义的GAN模型
    def __init__(self):
        super(CustomGenNet, self).__init__()
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

class GenModel:
    def __init__(self,args):
        #   init model
        self._args = args

        if self._args.gen_model == "aae":
            encoder = genmodels.aae.Encoder(self._args)
            decoder = genmodels.aae.Decoder(self._args)
            discriminator = genmodels.aae.Discriminator(self._args)               

        elif self._args.gen_model == "acgan":
            generator = genmodels.acgan.Generator(self._args)
            discriminator = genmodels.acgan.Discriminator(self._args)

        elif self._args.gen_model == "gan":
            generator = genmodels.gan.Generator(self._args)     
            discriminator = genmodels.gan.Discriminator(args) 

    def aaemodel(self):
        return 0
    
    def acganmodel(self):
        return 1

    def ganmodel(self):
        return 2 

class MixGenerate:
    r"""
        introduce this class
    """
    #   初始化
    def __init__(self, args, exp_result_dir, stylegan2ada_config_kwargs) -> None:

        #   initialize the parameters
        self._args = args
        self._exp_result_dir = exp_result_dir
        self._stylegan2ada_config_kwargs = stylegan2ada_config_kwargs

        if self._args.gen_model == "stylegan2ada":
            self._model = genmodels.stylegan2ada.MaggieStylegan2ada(self._args)                                                 #    实例化一个MaggieStylegan2ada类
        elif self._args.gen_model == "stylegan2":
            self._model = genmodels.stylegan2.MaggieStylegan2(self._args)                                                       #    实例化一个MaggieStylegan2ada类

        #   initialize the model
        if self._args.gen_network_pkl == None:           
            self._args.gen_network_pkl = self.__getpkl__()                                                                      #   训练好的模型路径赋值给 
            print("gen_network_pkl: " , self._args.gen_network_pkl)
        elif self._args.gen_network_pkl != None:
            print("gen_network_pkl: " , self._args.gen_network_pkl)

    def __getpkl__(self):
        genmodel_dict = ['gan', 'acgan', 'aae', 'vae','stylegan2','stylegan2ada']        
        if self._args.gen_model in genmodel_dict:
            gen_network_pkl = self.__getgenpkl__()
        else:
            gen_network_pkl = self.__getlocalpkl__()
        return gen_network_pkl

    def __getgenpkl__(self):
        if self._args.gen_model == "stylegan2ada":
            # self._model = genmodels.stylegan2ada.MaggieStylegan2ada(self._args)                                               #   实例化一个MaggieStylegan2ada类
            self._model.train(self._exp_result_dir, self._stylegan2ada_config_kwargs)
            snapshot_network_pkls = self._model.snapshot_network_pkls()                                                         #   返回一个snapshot_network_pkl列表
            snapshot_network_pkl = snapshot_network_pkls[-1]
        return snapshot_network_pkl

    def __getlocalpkl__(self)->"CustomGenNet":
        local_model_pkl = "abc test"
        return local_model_pkl




    #   建立流形映射
    def projectmain(self,cle_train_dataloader):
        #分成batch输入
        # print("cle_train_dataloader.dataset.__dict__:",cle_train_dataloader.dataset.__dict__)
        print("cle_train_dataloader.dataset.__dict__.keys():",cle_train_dataloader.dataset.__dict__.keys())
        # print("cle_train_dataloader.dataset.labels.shape:",cle_train_dataloader.dataset.labels.shape)
        print(" cle_train_dataloader.dataset.data[0].shape:", cle_train_dataloader.dataset.data[0].shape)   #torch.Size([28, 28])
        print("cle_train_dataloader..__dict__.keys():",cle_train_dataloader.__dict__.keys())

        # raise error

        if self._args.dataset =='cifar10' or self._args.dataset =='cifar100' or self._args.dataset =='kmnist':
            self.cle_x_train = cle_train_dataloader.dataset.data        #   读出来的数据是transformer前的
            self.cle_y_train = cle_train_dataloader.dataset.targets 
        elif self._args.dataset =='svhn':
            self.cle_x_train = cle_train_dataloader.dataset.data
            self.cle_y_train = cle_train_dataloader.dataset.labels            

        # print("cle_train_dataloader.dataset.__dict__:",cle_train_dataloader.dataset.__dict__)
        # print(cle_train_dataloader.dataset.classes)
        # print(cle_train_dataloader.dataset.class_to_idx)

        # raise error
        # print("self.cle_x_train.type:",type(self.cle_x_train))                          #   self.cle_x_train.type: <class 'numpy.ndarray'>
        # print("self.cle_x_train:",self.cle_x_train)                                     #   self.cle_x_train: [[[[ 59  62  63][ 43  46  45]...
        print("self.cle_x_train.shape:",self.cle_x_train.shape)                         #   self.cle_x_train.shape: (50000, 32, 32, 3)
        
        # print("self.cle_y_train.type:",type(self.cle_y_train))                          #   self.cle_y_train.type: <class 'list'>
        # print("self.cle_y_train:",self.cle_y_train)                                     #   self.cle_y_train: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7
        print("self.cle_y_train.len:",len(self.cle_y_train))                            #   self.cle_y_train.len: 50000

        # batch_num = int(np.ceil(len(self.cle_x_train) / float(self._args.batch_size)))           #   50099 /256=197
        sample_num = len(self.cle_x_train)
        batch_num = len(cle_train_dataloader)
        batch_size = self._args.batch_size
        print("sample_num:",sample_num)                                                             #   sample_num: 50000
        print("batch_num:",batch_num)                                                               #   batch_num: 196
        print("batch_size:",batch_size)                                                               #   batch_size: 256

        if self._args.mode == "project":
            if self._args.projected_dataset == None:
                cle_w_train = []
                cle_y_train = []                
                # print("cle_w_train.type:",type(cle_w_train))                                        #   cle_w_train.type: <class 'list'>
                # print("cle_y_train.type:",type(cle_y_train))                                        #   cle_y_train.type: <class 'list'>
                for batch_index in range(batch_num):                                                #   进入batch迭代 共有num_batch个batch
                    cle_x_trainbatch = self.cle_x_train[batch_index * batch_size : (batch_index + 1) * batch_size]
                    cle_y_trainbatch = self.cle_y_train[batch_index * batch_size : (batch_index + 1) * batch_size]                                                

                    # print("cle_x_trainbatch.type:",type(cle_x_trainbatch))                          #   cle_x_trainbatch.type: <class 'numpy.ndarray'>
                    # print("cle_x_trainbatch:",cle_x_trainbatch)                                     #   cle_x_trainbatch: [[[[ 59  62  63][ 43  46  45]
                    # print("cle_x_trainbatch.shape:",cle_x_trainbatch.shape)                         #   cle_x_trainbatch.shape: (256, 32, 32, 3)
                    # print("cle_y_trainbatch.type:",type(cle_y_trainbatch))                          #   cle_y_trainbatch.type: <class 'list'>
                    # print("cle_y_trainbatch:",cle_y_trainbatch)                                     #   cle_y_trainbatch: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7,
                    # print("cle_y_trainbatch.len:",len(cle_y_trainbatch))                            #   cle_y_trainbatch.len: 256

                    print(f"Projecting *{self._args.dataset}* {batch_index}/{batch_num} batch training sets...")                      #   projecting 00000031 image:
                    pro_w_trainbatch, pro_y_trainbatch = self.__batchproject__(batch_index,cle_x_trainbatch, cle_y_trainbatch)                                                                               #   numpy
                    
                    # print("pro_w_trainbatch.type:",type(pro_w_trainbatch))                              #   pro_w_trainbatch.type: <class 'torch.Tensor'>
                    # print("pro_w_trainbatch.shape:",pro_w_trainbatch.shape)                             #   pro_w_trainbatch.shape: torch.Size([32, 8, 512])
                    # print("pro_y_trainbatch.type:",type(pro_y_trainbatch))                              #   pro_y_trainbatch.type: <class 'torch.Tensor'>
                    # print("pro_y_trainbatch.shape:",pro_y_trainbatch.shape)                             #   pro_y_trainbatch.shape: torch.Size([32, 8])
                    # print("cle_w_train.type:",type(cle_w_train))                                        #   cle_w_train.type: <class 'list'>
                    # print("cle_y_train.type:",type(cle_y_train))                                        #   cle_y_train.type: <class 'list'>          
                    cle_w_train.append(pro_w_trainbatch)                  
                    cle_y_train.append(pro_y_trainbatch)
                    # print("cle_w_train.type:",type(cle_w_train))                                        #   cle_w_train.type: <class 'NoneType'>
                    # print("cle_y_train.type:",type(cle_y_train))                                        #   cle_y_train.type: <class 'NoneType'>

            else:
                raise Exception("参数 projected_dataset 不为空,无需投影！")

        cle_w_train_tensor = torch.stack(cle_w_train)                                                                         
        cle_y_train_tensor = torch.stack(cle_y_train)                                                                         

        self.cle_w_train = cle_w_train_tensor
        self.cle_y_train = cle_y_train_tensor

        print("self.cle_w_train.type:",type(self.cle_w_train))          #   torch
        print("self.cle_w_train.dtype:",self.cle_w_train.dtype)       
        print("self.cle_w_train.shape:",self.cle_w_train.shape)       

        print("self.cle_y_train.type:",type(self.cle_y_train))          #   torch
        print("self.cle_y_train.dtype:",self.cle_y_train.dtype)       
        print("self.cle_y_train.shape:",self.cle_y_train.shape)  

        print(f"Finished projecting {self._args.dataset} the whole {sample_num} samples!")

    def interpolatemain(self):

        mix_w_train, mix_y_train = self.interpolate()                                                                           #   numpy
        
        self.mix_w_train = mix_w_train
        self.mix_y_train = mix_y_train

        print("self.mix_w_train.type:",type(self.mix_w_train))          #   torch
        print("self.mix_w_train.dtype:",self.mix_w_train.dtype)       
        print("self.mix_w_train.shape:",self.mix_w_train.shape)       

        print("self.mix_y_train.type:",type(self.mix_y_train))          #   torch
        print("self.mix_y_train.dtype:",self.mix_y_train.dtype)       
        print("self.mix_y_train.shape:",self.mix_y_train.shape)  
        
        print(f"Finished interpolate {self._args.dataset} {len(self.mix_w_train)} samples!")
        self.generatemain()
        print(f"Finished generate {self._args.dataset} {len(self.mix_w_train)} interpolated samples!")
        
    def generatemain(self):
        generated_x_train, generated_y_train = self.generate()
        self.generated_x_train = generated_x_train
        self.generated_y_train = generated_y_train
        print(f"Finished generate {self._args.dataset} {len(self.mix_w_train)} interpolated samples!")

    def mixgenerate(self,cle_train_dataloader) -> "tensor" :
     
        # print(f"Attributes of *{self._args.dataset}* cle_train_dataset:{cle_train_dataloader.dataset.__dict__.keys()}")

        # self.cle_x_train = cle_train_dataloader.dataset.data
        # self.cle_y_train = cle_train_dataloader.dataset.targets

        # print("self.cle_x_train.type:",type(self.cle_x_train))
        # print("self.cle_x_train:",self.cle_x_train)        
        # print("self.cle_y_train.type:",type(self.cle_y_train))
        # print("self.cle_y_train:",self.cle_y_train)        

        # if self._args.projected_dataset == None:
        #     #   投影
        #     cle_w_train, cle_y_train = self.project()                                                                               #   numpy
        #     self.cle_w_train = cle_w_train
        #     self.cle_y_train = cle_y_train
        # else:
        #     print("使用事先准备好的投影!")

        if self._args.mix_dataset == None:     
            #   混合
            mix_w_train, mix_y_train = self.interpolate()                                                                           #   numpy
            self.mix_w_train = mix_w_train
            self.mix_y_train = mix_y_train

            #   生成
            generated_x_train, generated_y_train = self.generate()
            self.generated_x_train = generated_x_train
            self.generated_y_train = generated_y_train
        else:
            print("使用事先准备好的插值样本!")

    def generatedset(self):
        if self._args.mix_dataset == None:
            return self.generated_x_train,self.generated_y_train
        else:           
            generated_x_train, generated_y_train = self.getmixset(self._args.mix_dataset)
            self.generated_x_train = generated_x_train
            self.generated_y_train = generated_y_train
            return self.generated_x_train,self.generated_y_train

    def getmixset(self,mix_dataset_path):
        
        file_dir=os.listdir(mix_dataset_path)
        file_dir.sort()
        # print("file_dir:",file_dir)                                                                               #   file_dir: ['00000000-adv-3-cat.npz', '00000000-adv-3-cat.png',
        # print("file_dir[0][-9:-4] :",file_dir[0][-9:-4])                                                            #   file_dir[0][9:12] : adv
        # print("file_dir[0][-9:4] :",file_dir[0][-9:-4])                                                            #   file_dir[0][9:12] : adv
        
        # filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz'] 
        img_filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[-9:-4] == 'image']           
        label_filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[-9:-4] == 'label']           
        
        # print("img_filenames:",img_filenames)                                                                   #   img_filenames: ['00000000-9-truck+6-frog-mixed-image.npz']
        print("label_filenames[0]:",label_filenames[0])                                     #   label_filenames[0]: 00000000-6-frog+00000001-9-truck-mixed_label.npz
        print("label_filenames.len:",len(label_filenames))                                  #   label_filenames.len: 42358
        
        mix_xset_tensor = []

        print("test here")
        for _, img_filename in enumerate(img_filenames):
            print("img_filename:",img_filename)
            print('in..........')
            mix_img_npz_path = os.path.join(mix_dataset_path,img_filename)
            # print("mix_img_npz_path:",mix_img_npz_path)
            load_mix_img = np.load(mix_img_npz_path)['w']            
            load_mix_img = torch.tensor(load_mix_img)
            # print("load_mix_img.shape",load_mix_img.shape)
            mix_xset_tensor.append(load_mix_img)
            # raise error

        mix_yset_tensor = []
        for _, lab_filename in enumerate(label_filenames):
         
            mix_lab_npz_path = os.path.join(mix_dataset_path,lab_filename)

            load_mix_lab = np.load(mix_lab_npz_path)['w']            
            load_mix_lab = torch.tensor(load_mix_lab)
            mix_yset_tensor.append(load_mix_lab)

        print("mix_xset_tensor.len",len(mix_xset_tensor))
        print("mix_xset_tensor[0]",mix_xset_tensor[0])          #0 
        # raise error
        mix_xset_tensor = torch.stack(mix_xset_tensor)                                                                         
        mix_yset_tensor = torch.stack(mix_yset_tensor)   

        # print("mix_xset_tensor.type:",type(mix_xset_tensor))                                                    #  mix_xset_tensor[0][0]: tensor([[ 1.0653,  0.9195,  0.6705,  ...,  0.1975,  0.2704,  0.0212]                                                     
        # print("mix_xset_tensor.shape:",mix_xset_tensor.shape)                                                   #  mix_xset_tensor.shape: torch.Size([1, 3, 32, 32]) 
        # print("mix_xset_tensor[0][0]:",mix_xset_tensor[0][0])                                                   #  
        
        
        # print("mix_yset_tensor.type:",type(mix_yset_tensor))                                                    # mix_yset_tensor: tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3126, 0.0000, 0.0000, 0.6874]])                              
        # print("mix_yset_tensor.shape:",mix_yset_tensor.shape)                                                   #  mix_yset_tensor.shape: torch.Size([1, 10])
        # print("mix_yset_tensor:",mix_yset_tensor)                                                               #   

        # raise error

        return mix_xset_tensor.cuda(), mix_yset_tensor.cuda()

    def project(self):
        if self._args.gen_model == "stylegan2ada":

            if self._args.viewdataset_path == None:                                                 #   从内存dataloader中加载tensor
                self._model.project(self._exp_result_dir,self.cle_x_train,self.cle_y_train)         #   调用stylegan2ada的project
            elif self._args.viewdataset_path != None:
                self._model.project(self._exp_result_dir)

            cle_w_train, cle_y_train = self._model.wyset()                                                                      #   tensor的list， int的list

            cle_w_train = torch.stack(cle_w_train)                                                                              #   torch.Tensor GPU Tensor
            print('pro_w_train.shape:',cle_w_train.shape)                                                                       #   pro_w_train.shape: torch.Size([4, 8, 512])
            
            cle_y_train = torch.stack(cle_y_train)                                                                              #   torch.Tensor GPU Tensor
            print('pro_y_train.shape:',cle_y_train.shape)                                                                       #   pro_y_train.shape: torch.Size([4, 8])
    
        return cle_w_train,cle_y_train  
                                                                                            
    def __batchproject__(self, batch_index, cle_x_trainbatch, cle_y_trainbatch):
        if self._args.gen_model == "stylegan2ada":
            self._model.project(self._exp_result_dir,cle_x_trainbatch,cle_y_trainbatch,batch_index)         #   调用stylegan2ada的project
            cle_w_train, cle_y_train = self._model.wyset()                                                                      #   tensor的list， int的list
            cle_w_train = torch.stack(cle_w_train)                                                                              #   torch.Tensor GPU Tensor
            print('pro_w_train.shape:',cle_w_train.shape)     #                                                                   #   pro_w_train.shape: torch.Size([4, 8, 512])
            cle_y_train = torch.stack(cle_y_train)                                                                              #   torch.Tensor GPU Tensor
            print('pro_y_train.shape:',cle_y_train.shape)                                                                       #   pro_y_train.shape: torch.Size([4, 8])
        return cle_w_train,cle_y_train  

    def interpolate(self):
        if self._args.gen_model == "stylegan2ada":
            
            if self._args.projected_dataset == None:
                self._model.interpolate(self._exp_result_dir, self.cle_w_train, self.cle_y_train)
            else:
                self._model.interpolate(self._exp_result_dir)                                                                     #   测试读取本地npz投影

            mix_w_train, mix_y_train = self._model.mixwyset()                                                                   #   tensor的list， int的list
               
            mix_w_train = torch.stack(mix_w_train)                                                                              #   torch.Tensor GPU Tensor       
            print('mix_w_train.shape:',mix_w_train.shape)                                                                       #   mix_w_train.shape: torch.Size([3, 8, 512])
        
            mix_y_train = torch.stack(mix_y_train)                                                                              #   torch.Tensor GPU Tensor       
            print('mix_y_train.shape:',mix_y_train.shape)                                                                       #   mix_y_train.shape: torch.Size([3, 8, 10])

        return mix_w_train, mix_y_train                        

    def generate(self):
        if self._args.gen_model == "stylegan2ada":
            
            if self._args.mixed_dataset ==None:
                print("无 mix dataset path")
                self._model.generate(self._exp_result_dir, self.mix_w_train, self.mix_y_train)
            else:
                print("有 mix dataset path")
                self._model.generate(self._exp_result_dir)

            generated_x_train, generated_y_train = self._model.genxyset() 
            
            generated_x_train = torch.stack(generated_x_train)                                                                  #   torch.Tensor GPU Tensor           
            print('generated_x_train.shape:',generated_x_train.shape)                                                           #   generated_x_train.shape: torch.Size([3, 3, 32, 32])
            
            generated_y_train = torch.stack(generated_y_train)                                                                  #   torch.Tensor GPU Tensor   
            print('generated_y_train.shape:',generated_y_train.shape)                                                           #   generated_y_train.shape: torch.Size([3, 10])

        return generated_x_train, generated_y_train    
    