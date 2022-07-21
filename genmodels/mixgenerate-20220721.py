"""
Author: maggie
Date:   2021-06-22
Place:  Xidian University
@copyright
"""

from logging import error, exception
from random import sample
from PIL.Image import RASTERIZE
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
            # print("gen_network_pkl: " , self._args.gen_network_pkl)
        # elif self._args.gen_network_pkl != None:
        #     print("\n")

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
        else:
            print("without gen model")
        return snapshot_network_pkl

    def __getlocalpkl__(self)->"CustomGenNet":
        local_model_pkl = "abc test"
        return local_model_pkl

    #   建立流形映射
    def projectmain(self, cle_train_dataloader):
        #分成batch输入
        print("self._args.dataset:",self._args.dataset)
        # print("cle_train_dataloader.dataset.__dict__:",cle_train_dataloader.dataset.__dict__) #    ('/home/data/ImageNet/train/n13052670/n13052670_9998.JPEG', 8)

        # print("cle_train_dataloader.dataset.__dict__.keys():",cle_train_dataloader.dataset.__dict__.keys())
        # cle_train_dataloader.dataset.__dict__.keys(): dict_keys(['root', 'loader', 'extensions', 'classes', 'class_to_idx', 'samples', 'targets', 'transform', 'target_transform', 'imgs'])

        # print("cle_train_dataloader.dataset.classes:",cle_train_dataloader.dataset.classes)             #   cle_train_dataloader.dataset.classes: None

        # print("cle_train_dataloader.dataset.class_to_idx:",cle_train_dataloader.dataset.class_to_idx)
        #   cle_train_dataloader.dataset.class_to_idx: 
        #   {
        #   'n01514668': 1, 'n01514859': 1, 'n01518878': 1, 'n01530575': 1, 'n01531178': 1, 'n01532829': 1, 
        #   'n02085620': 0, 'n02085782': 0, 'n02085936': 0, 'n02086079': 0, 'n02086240': 0, 'n02086646': 0, 
        #   'n02125311': 5, 'n02127052': 5, 'n02128385': 5, 'n02128757': 5, 'n02128925': 5, 'n02129165': 5, 
        #   'n02165105': 2, 'n02165456': 2, 'n02167151': 2, 'n02168699': 2, 'n02169497': 2, 'n02172182': 2, 
        #   'n02484975': 3, 'n02486261': 3, 'n02486410': 3, 'n02487347': 3, 'n02488291': 3, 'n02488702': 3, 
        #   'n02701002': 4, 'n02814533': 4, 'n02930766': 4, 'n03100240': 4, 'n03594945': 4, 'n03670208': 4,
        #   'n02951358': 9, 'n03344393': 9, 'n03447447': 9, 'n03662601': 9, 'n04273569': 9, 'n04612504': 9,
        #   'n03345487': 6, 'n03417042': 6, 'n03796401': 6, 'n03930630': 6, 'n03977966': 6, 'n04461696': 6,  
        #   'n07742313': 7, 'n11879895': 7, 'n12144580': 7, 'n12267677': 7, 'n12620546': 7, 'n12768682': 7, 
        #   'n12985857': 8, 'n12998815': 8, 'n13037406': 8, 'n13040303': 8, 'n13044778': 8, 'n13052670': 8}

        # print("cle_train_dataloader.dataset.targets:",cle_train_dataloader.dataset.targets)        #   

        # print("cle_train_dataloader.dataset.targets.len:",len(cle_train_dataloader.dataset.targets))        #   cle_train_dataloader.dataset.targets.len: 77237 即mixed10训练集中的样本总数

        # print("cle_train_dataloader.dataset.targets.samples:",cle_train_dataloader.dataset.samples)        

        # print("cle_train_dataloader.dataset.targets.transform:",cle_train_dataloader.dataset.transform)        # 变换函数 cle_train_dataloader.dataset.targets.transform: Compose(Resize(size=256, 256), interpolation=bilinear)    ToTensor())


        # print("cle_train_dataloader.dataset.target_transform:",cle_train_dataloader.dataset.target_transform)   #   cle_train_dataloader.dataset.target_transform: None

        # print("cle_train_dataloader.dataset.imgs:",cle_train_dataloader.dataset.imgs)   #   ('/home/data/ImageNet/train/n13052670/n13052670_9998.JPEG', 8)

        # print("cle_train_dataloader.dataset.imgs[0][0]:",cle_train_dataloader.dataset.imgs[0][0])           #   cle_train_dataloader.dataset.imgs[0][0]: /home/data/ImageNet/train/n01514668/n01514668_10004.JPEG

        # print("cle_train_dataloader.dataset.loader:",cle_train_dataloader.dataset.loader) # 加载函数


        # print("cle_train_dataloader.__dict__.keys():",cle_train_dataloader.__dict__.keys())
        #   cle_train_dataloader.__dict__.keys(): dict_keys(['dataset', 'num_workers', 'prefetch_factor', 'pin_memory', 'timeout', 'worker_init_fn', '_DataLoader__multiprocessing_context', '_dataset_kind', 'batch_size', 'drop_last', 'sampler', 'batch_sampler', 'generator', 'collate_fn', 'persistent_workers', '_DataLoader__initialized', '_IterableDataset_len_called', '_iterator'])

        # print("cle_train_dataloader._dataset_kind:",cle_train_dataloader._dataset_kind)

        # for idx, (img, lab) in enumerate(cle_train_dataloader):
        #     print("img.shape:",img.shape)               #   img.shape: torch.Size([32, 3, 256, 256])
        #     print("img.dtype:",img.dtype)               #   img.dtype: torch.float32
        #     print("img:",img)                           #   img: tensor([[[[0.2941, 0.2392, 0.2431,  ..., 0.6196, 0.6275, 0.6235],
        #     print("lab.dtype:",lab.dtype)               #   lab.dtype: torch.int64
        #     print("lab:",lab)                           #   lab: tensor([3, 7, 0, 6, 4, 4, 9, 0, 3, 7, 1, 2, 1, 4, 4, 0, 8, 8, 1, 3, 2, 9, 5, 3,7, 9, 3, 3, 3, 3, 2, 5]
        #     raise error


        # print("custom_dataset.__dict__.keys()",custom_dataset.__dict__.keys())
        # custom_dataset.__dict__.keys() dict_keys(['ds_name', 'data_path', 'num_classes', 'mean', 'std', 'transform_train', 'transform_test', 'custom_class', 'label_mapping', 'custom_class_args'])

        # raise error

        if self._args.dataset =='cifar10' or self._args.dataset =='cifar100' or self._args.dataset =='kmnist':
            self.cle_x_train = cle_train_dataloader.dataset.data        #   读出来的数据是transformer前的
            self.cle_y_train = cle_train_dataloader.dataset.targets 
        elif self._args.dataset =='svhn' or self._args.dataset =='stl10':
            self.cle_x_train = cle_train_dataloader.dataset.data
            self.cle_y_train = cle_train_dataloader.dataset.labels   
        if  self._args.dataset =='imagenetmixed10':
            self.cle_y_train = cle_train_dataloader.dataset.targets
            # self.cle_x_train = cle_train_dataloader.dataset.targets

        # print("cle_train_dataloader.dataset.__dict__:",cle_train_dataloader.dataset.__dict__)
        # print(cle_train_dataloader.dataset.classes)
        # print(cle_train_dataloader.dataset.class_to_idx)

        # raise error
        # print("self.cle_x_train.type:",type(self.cle_x_train))                          #   self.cle_x_train.type: <class 'numpy.ndarray'>
        # print("self.cle_x_train:",self.cle_x_train)                                     #   self.cle_x_train: [[[[ 59  62  63][ 43  46  45]...
        # print("self.cle_x_train.shape:",self.cle_x_train.shape)                         #   self.cle_x_train.shape: (50000, 32, 32, 3)
        
        # print("self.cle_y_train.type:",type(self.cle_y_train))                          #   self.cle_y_train.type: <class 'list'>
        # print("self.cle_y_train:",self.cle_y_train)                                     #   self.cle_y_train: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7
        # print("self.cle_y_train.len:",len(self.cle_y_train))                            #   self.cle_y_train.len: 50000

        # batch_num = int(np.ceil(len(self.cle_x_train) / float(self._args.batch_size)))           #   50099 /256=197
        sample_num = len(self.cle_y_train)
        batch_num = len(cle_train_dataloader)
        batch_size = self._args.batch_size
        print("sample_num:",sample_num)                                                                 #   sample_num: 77237
        print("batch_num:",batch_num)                                                                   #   batch_num: 2414
        print("batch_size:",batch_size)                                                                 #   batch_size: 32


        if self._args.mode == "project":
            if self._args.projected_dataset == None:

                if self._args.dataset != 'imagenetmixed10':
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

                        print(f"Projecting *{self._args.dataset}* {batch_index}/{batch_num} batch data sets...")                      #   projecting 00000031 image:
                        pro_w_trainbatch, pro_y_trainbatch = self.__batchproject__(batch_index,cle_x_trainbatch, cle_y_trainbatch)                                 #   numpy
                        
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
               
                elif self._args.dataset == 'imagenetmixed10':
                    cle_w_train = []
                    cle_y_train = [] 

                    for batch_index in range(batch_num):
                        print("batch_index:",batch_index)                     # batch_index: 0

                        for batch_idx, (imgs, labs) in enumerate(cle_train_dataloader):
                            print("batch_idx:",batch_idx)
 
                            if batch_idx == batch_index:
                                # print("img.shape:",img.shape)               #   img.shape: torch.Size([32, 3, 256, 256])
                                # print("img.dtype:",img.dtype)               #   img.dtype: torch.float32
                                # print("img:",img)                           #   img: tensor([[[[0.2941, 0.2392, 0.2431,  ..., 0.6196, 0.6275, 0.6235],

                                # print("image index:",idx)
                                # print("img.shape",img.shape)
                                imgs.reshape(-1, 3, 256, 256)
                                # print("imgs.shape",imgs.shape)                      #   imgs.shape torch.Size([32, 3, 256, 256])

                                imgs = imgs.numpy()

                                # print("imgs.shape",imgs.shape)                      #   imgs.shape (32, 3, 256, 256)
                                imgs = imgs.transpose([0, 2, 3, 1])               #   NCHW -> NHWC
                                # print("imgs.shape:",imgs.shape)                   #   imgs.shape: (32, 256, 256, 3)

                                # print("imgs.dtype",imgs.dtype)                    #   imgs.dtype float32
                                # print("imgs:",imgs)

                                imgs = (imgs*255).astype(np.uint8)
                                # print("imgs.dtype",imgs.dtype)                    #   imgs.dtype uint8
                                # print("imgs:",imgs)

                                cle_x_trainbatch = imgs
                                cle_y_trainbatch = labs.numpy().tolist()
                                # print("cle_x_trainbatch.shape:",cle_x_trainbatch.shape)
                                # print("cle_y_trainbatch.len:",len(cle_y_trainbatch))

                                # print("cle_x_trainbatch.type:",type(cle_x_trainbatch))                          #   cle_x_trainbatch.type: <class 'numpy.ndarray'>
                                # print("cle_x_trainbatch:",cle_x_trainbatch)                                     #   cle_x_trainbatch: [[[[ 44  48  34] [ 99 121  58]
                                # print("cle_x_trainbatch.shape:",cle_x_trainbatch.shape)                         #   cle_x_trainbatch.shape: (32, 256, 256, 3)
                                # print("cle_y_trainbatch.type:",type(cle_y_trainbatch))                          #   cle_y_trainbatch.type: <class 'list'>
                                # print("cle_y_trainbatch:",cle_y_trainbatch)                                     #   cle_y_trainbatch: [3, 4, 1, 6, 3, 6, 7, 8, 7, 3, 5,
                                # print("cle_y_trainbatch.len:",len(cle_y_trainbatch))                            #   cle_y_trainbatch.len: 32

                                # raise error
                                print(f"Projecting *{self._args.dataset}* {batch_index}/{batch_num} batch data sets...")                      #   projecting 00000031 image:
                                pro_w_trainbatch, pro_y_trainbatch = self.__batchproject__(batch_index,cle_x_trainbatch, cle_y_trainbatch)                 #   numpy
                                
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
            print(f"load mixed sampels from {self._args.mix_dataset}")

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
        # 00000000-2-+5-5-mixed-image.npz
        # 00000000-2-2+5-5-mixed-label.npz

        img_filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[-9:-4] == 'image']           
        label_filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[-9:-4] == 'label']           
        
        select_mix_num = int( self._args.aug_mix_rate * self._args.aug_num )
        print(f"共使用 {select_mix_num} 个混合样本")
        
        mix_xset_tensor = []
        for miximg_index, img_filename in enumerate(img_filenames):
            if miximg_index < select_mix_num:
                mix_img_npz_path = os.path.join(mix_dataset_path,img_filename)
                load_mix_img = np.load(mix_img_npz_path)['w']            
                load_mix_img = torch.tensor(load_mix_img)
                mix_xset_tensor.append(load_mix_img)

        mix_yset_tensor = []
        for mixy_index, lab_filename in enumerate(label_filenames):
            if mixy_index < select_mix_num:  
                mix_lab_npz_path = os.path.join(mix_dataset_path,lab_filename)
                load_mix_lab = np.load(mix_lab_npz_path)['w']            
                load_mix_lab = torch.tensor(load_mix_lab)
                mix_yset_tensor.append(load_mix_lab)

        mix_xset_tensor = torch.stack(mix_xset_tensor)                                                                         
        mix_yset_tensor = torch.stack(mix_yset_tensor)  

        # print("mix_xset_tensor.shape:",mix_xset_tensor) 
        # print("mix_yset_tensor.shape:",mix_yset_tensor) 

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
            
            # print("self._args.projected_dataset:",self._args.projected_dataset )    #   self._args.projected_dataset: None
            # print("self._args.projected_w1:",self._args.projected_w1)
            
            """
            self._args.projected_w1: None
            self.cle_w_train.shape: torch.Size([4, 8, 512])
            self.cle_y_train.shape: torch.Size([4, 8, 10])
            """
            if self._args.defense_mode == 'rmt':
                # print("self.cle_w_train.shape:",self.cle_w_train.shape)
                # print("self.cle_y_train.shape:",self.cle_y_train.shape)
                self._model.interpolate(self._exp_result_dir, self.cle_w_train, self.cle_y_train)
                mix_w_train, mix_y_train = self._model.mixwyset()   #   当采用整个batch采样时，返回的直接就是tensor,不需要torch.stack从list转tensor

            else:
                if self._args.projected_dataset is None and self._args.projected_w1 is None :        # 从内存中加载待投影x,y
                    self._model.interpolate(self._exp_result_dir, self.cle_w_train, self.cle_y_train)
                
                elif self._args.projected_dataset is not None or self._args.projected_w1 is not None:   #从本地npz数据集中加载待投影x,y   或 从本地2个npz文件中加载待投影x,y 
                    self._model.interpolate(self._exp_result_dir)                                                                     #   测试读取本地个别npz投影

                #   读取mix样本
                mix_w_train, mix_y_train = self._model.mixwyset()                                                                   #   tensor的list， int的list

                # mix_w_train = torch.stack(mix_w_train)                                                                              #   torch.Tensor GPU Tensor             
                # mix_y_train = torch.stack(mix_y_train)                                                                              #   torch.Tensor GPU Tensor       

                mix_w_train = torch.stack(mix_w_train).cuda()                                                                              #   torch.Tensor GPU Tensor             
                mix_y_train = torch.stack(mix_y_train).cuda()                                                                              #   torch.Tensor GPU Tensor       

                # print('mix_w_train:',mix_w_train)                                                                       #   mix_w_train.shape: torch.Size([3, 8, 512])            
                # print('mix_y_train:',mix_y_train)                                                                       #   mix_y_train.shape: torch.Size([3, 8, 10])
                # print('mix_w_train.shape:',mix_w_train.shape)                                           #   mix_w_train.shape: torch.Size([3, 8, 512])            
                # print('mix_y_train.shape:',mix_y_train.shape)                                           #   mix_y_train.shape: torch.Size([3, 8, 10])

                """
                mix_w_train.shape: torch.Size([4, 8, 512])
                mix_y_train.shape: torch.Size([4, 8, 10])
                """

            return mix_w_train, mix_y_train                        

    def generate(self):
        if self._args.gen_model == "stylegan2ada":

            if self._args.defense_mode == 'rmt':

                self._model.generate(self._exp_result_dir, self.mix_w_train, self.mix_y_train)
                generated_x_train, generated_y_train = self._model.genxyset()                                                                    #   当采用整个batch采样时，返回的直接就是tensor,不需要torch.stack从list转tensor
                # print('generated_x_train.shape:',generated_x_train.shape)                                  #   generated_x_train.shape: torch.Size([3, 3, 32, 32])
                # print('generated_y_train.shape:',generated_y_train.shape)                                  #   generated_y_train.shape: torch.Size([3, 10])
                               
                # raise error
            else:

                if self._args.mixed_dataset ==None:
                    
                    if self._args.generate_seeds is not None:
                        self._model.generate(self._exp_result_dir)
                    else:
                        # print("maggie flag 20210920")
                        self._model.generate(self._exp_result_dir, self.mix_w_train, self.mix_y_train) #    都从这里进入

                elif self._args.mixed_dataset !=None:
                    print("有 mix dataset path")
                    self._model.generate(self._exp_result_dir)

                generated_x_train, generated_y_train = self._model.genxyset() 
                
                generated_x_train = torch.stack(generated_x_train)                                                                  #   torch.Tensor GPU Tensor           

                generated_y_train = torch.stack(generated_y_train)                                                                  #   torch.Tensor GPU Tensor   
                # print('generated_x_train:',generated_x_train)                                  #   generated_x_train.shape: torch.Size([3, 3, 32, 32])
                # print('generated_y_train:',generated_y_train)                                  #   generated_y_train.shape: torch.Size([3, 10])
                
                # print('generated_x_train.shape:',generated_x_train.shape)                                  #   generated_x_train.shape: torch.Size([3, 3, 32, 32])
                # print('generated_y_train.shape:',generated_y_train.shape)                                  #   generated_y_train.shape: torch.Size([3, 10])

        return generated_x_train, generated_y_train    
    