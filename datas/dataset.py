"""
Author: maggie
Date:   2021-06-18
Place:  Xidian University
@copyright
"""

import os
from torchvision.transforms import transforms
import torchvision.datasets
import numpy as np
import copy
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
import robustness.datasets


class MaggieMNIST(torchvision.datasets.MNIST):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace mnist data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets

        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 28, 28)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 1))  # convert to HWC
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    
    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment mnist data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
      
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 28, 28)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieKMNIST(torchvision.datasets.KMNIST):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace kmnist data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 28, 28)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 1))  # convert to HWC
       
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data


    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment kmnist data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 28, 28)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 1))  # convert to HWC
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets
 
        self.data = data


class MaggieCIFAR10(torchvision.datasets.CIFAR10):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace cifar10 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        print('constraucting adv cifar10 testset')
        data = self.data
        # targets = self.targets

        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC

        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment cifar10 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data            #   没有这行时，替换操作前后，self.data值不变


class MaggieCIFAR100(torchvision.datasets.CIFAR100):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace cifar100 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
        
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data


    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment cifar100 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
      
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieImageNet(torchvision.datasets.ImageNet):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace imagenet data and targets with rep_xndarray,rep_yndarray ")

    # change this function
    def __getrepdataset__(self):

        data = self.data    #   ImageNet has no attribute named data
        # targets = self.targets
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 256, 256)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment imagenet data and targets with aug_xndarray,aug_yndarray ")
    
    # change this function
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 256, 256)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieLSUN(torchvision.datasets.LSUN):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace lsun data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 256, 256)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment lsun data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 256, 256)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
     
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieSVHN(torchvision.datasets.SVHN):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace svhn data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray

        self.data = data

    def augdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment svhn data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data    


class MaggieSTL10(torchvision.datasets.STL10):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace stl10 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        # targets = self.targets
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 96, 96)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        # targets = self._rep_yndarray

        self.data = data

    def augdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment stl10 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 96, 96)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  # convert to HWC
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data

#-------------------------------------------------------

class MaggieDataset:
    def __init__(self,args, custom_traindataset = None, custom_testdataset =None) -> None:
        print(f'initilize the dataset loading parameters')
        self._args = args 
        if custom_traindataset == None:
            # print("dataset from pytorch")
            self._traindataset = self.__loadtraindataset__()  
            self._testdataset = self.__loadtestdataset__() 
        else:
            # print("dataset from custom")
            self._traindataset = custom_traindataset
            self._testdataset = custom_testdataset

    def traindataset(self):
        return self._traindataset
    
    def testdataset(self):
        return self._testdataset

    def __loadtraindataset__(self):
        if self._args.dataset == 'mnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size            

            os.makedirs("/home/data/maggie/mnist", exist_ok=True)
            train_dataset = MaggieMNIST(                                             
                "/home/data/maggie/mnist",
                train=True,                                             #   从training.pt创建数据集
                download=True,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset

        elif self._args.dataset == 'kmnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size            
            
            os.makedirs("/home/data/maggie/kmnist", exist_ok=True)
            train_dataset = MaggieKMNIST(                                            
                "/home/data/maggie/kmnist",
                train=True,                                             #   从training.pt创建数据集
                download=True,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            ) 
            return train_dataset

        elif self._args.dataset == 'cifar10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            print(f'load cifar10 dataset')
            os.makedirs("/home/data/maggie/cifar10", exist_ok=True)
            train_dataset = MaggieCIFAR10(                                             
                "/home/data/maggie/cifar10",
                train=True,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            # print('train_dataset:',train_dataset)
            # print('train_dataset.__dict__',train_dataset.__dict__)
            return train_dataset

        elif self._args.dataset == 'cifar100':
            os.makedirs("/home/data/maggie/cifar100", exist_ok=True)  
            
            train_dataset = MaggieCIFAR100(                                            
                "/home/data/maggie/cifar100",
                train=True,                                             #   从training.pt创建数据集
                download=True,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(self._args.img_size), 
                        transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            return train_dataset

        elif self._args.dataset == 'imagenet':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size # 256

            os.makedirs("/home/data/ImageNet", exist_ok=True)
            train_dataset = MaggieImageNet(                                             
                "/home/data/ImageNet",
                split='train',
                download=False,
                transform=transforms.Compose(                               #   组合多个图像变换
                    [
                        transforms.Resize(crop_size),                       #   通过Resize(size, interpolation=Image.BILINEAR)函数将输入的图像转换为期望的尺寸，此处期望的输出size=img_size
                        transforms.CenterCrop(crop_size),                    #   基于给定输入图像的中心，按照期望的尺寸（img_size）裁剪图像！！！解决了ImageNet数据集中个别样本size不符合3x256x256引发的问题
                        transforms.ToTensor(),                                  #   将PIL图像或者ndArry数据转换为tensor
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]     
                ),
            ) 
            return train_dataset

        elif self._args.dataset == 'imagenetmixed10':
            # in_path = "/home/data/ImageNet"             #   ImageNet解压后的train和val所在的目录
            # in_info_path = "/home/data/ImageNet/info"
            # in_path = "/root/autodl-tmp/maggie/data/ImageNet"             #   ImageNet解压后的train和val所在的目录
            # in_info_path = "/root/autodl-tmp/maggie/data/ImageNet/info"
           
            in_path = "/home/huan1932/data/ImageNet"             #   ImageNet解压后的train和val所在的目录
            in_info_path = "/home/huan1932/data/ImageNet/info"
                      
            in_hier = ImageNetHierarchy(in_path, in_info_path)                  

            superclass_wnid = common_superclass_wnid('mixed_10')            # group name: mixed_10
            class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

            # num_workers =4
            # batch_size =1
            custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
            print("custom_dataset.__dict__.keys()",custom_dataset.__dict__.keys())
            # custom_dataset.__dict__.keys() dict_keys(['ds_name', 'data_path', 'num_classes', 'mean', 'std', 'transform_train', 'transform_test', 'custom_class', 'label_mapping', 'custom_class_args'])
            # train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, batch_size=batch_size)
            train_dataset = custom_dataset

            # if self._args.train_mode != 'cla-train':
            #     custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
            #     print("custom_dataset.__dict__.keys()",custom_dataset.__dict__.keys())
            #     # custom_dataset.__dict__.keys() dict_keys(['ds_name', 'data_path', 'num_classes', 'mean', 'std', 'transform_train', 'transform_test', 'custom_class', 'label_mapping', 'custom_class_args'])
            #     # train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, batch_size=batch_size)

            # elif self._args.train_mode == 'cla-train':
            #     Maggie_TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
            #             transforms.Resize((256, 256)),
            #             transforms.CenterCrop(224),
            #             transforms.ToTensor()
            #         ])
            #     Maggie_TEST_TRANSFORMS_IMAGENET = transforms.Compose([
            #             transforms.Resize((256, 256)),
            #             transforms.CenterCrop(224),
            #             transforms.ToTensor()
            #         ])                    

            #     custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges, transform_train = Maggie_TRAIN_TRANSFORMS_IMAGENET, transform_test= Maggie_TEST_TRANSFORMS_IMAGENET)

            # train_dataset = custom_dataset

                # raise Exception("maggie error")
            return train_dataset

        elif self._args.dataset == 'lsun':
            os.makedirs("/home/data/maggie/lsun/20210413", exist_ok=True)
            train_dataset =MaggieLSUN(                                             
                "/home/data/maggie/lsun/20210413",
                #classes='train',   #   部分class由于无法解压缩包，不能训练
                #classes=['church_outdoor_train','classroom_train','tower_train'],   #   部分class由于无法解压缩包，不能训练
                classes=['church_outdoor_train','classroom_train','tower_train'],
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(self._args.img_size),                       
                        transforms.CenterCrop(self._args.img_size),                    
                        transforms.ToTensor(),                                  
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]     
                ),
                target_transform = None
            ) 
            return train_dataset
        
        elif self._args.dataset == 'svhn':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/maggie/svhn", exist_ok=True)
            train_dataset = MaggieSVHN(                                             
                "/home/data/maggie/svhn",
                split='train',                                             #   从training.pt创建数据集
                download=True,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset            

        elif self._args.dataset == 'stl10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size    

            os.makedirs("/home/data/maggie/stl10", exist_ok=True)
            train_dataset = MaggieSTL10(                                             
                "/home/data/maggie/stl10",
                split='train',                                             #   从training.pt创建数据集
                download=True,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset

    def __loadtestdataset__(self):
        if self._args.dataset == 'mnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size    

            os.makedirs("/home/data/maggie/mnist", exist_ok=True)
            test_dataset = MaggieMNIST(                                             
                "/home/data/maggie/mnist",
                train=False,                                             #   从training.pt创建数据集
                download=False,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'kmnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size               
            os.makedirs("/home/data/maggie/kmnist", exist_ok=True)
            test_dataset = MaggieKMNIST(                                          
                "/home/data/maggie/kmnist",
                train=False,                                             #   从training.pt创建数据集
                download=False,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'cifar10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/maggie/cifar10", exist_ok=True)
            test_dataset = MaggieCIFAR10(                                             
                "/home/data/maggie/cifar10",
                train=False,                                             #   从training.pt创建数据集
                download=False,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'cifar100':
            os.makedirs("/home/data/maggie/cifar100", exist_ok=True)
            test_dataset = MaggieCIFAR100(                                             
                "/home/data/maggie/cifar100",
                train=False,                                             #   从training.pt创建数据集
                download=False,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(self._args.img_size), 
                        transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'imagenet':

            os.makedirs("/home/data/ImageNet", exist_ok=True)

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size # 256
                # crop_size = 1024

            test_dataset = MaggieImageNet(                                             
                "/home/data/ImageNet",
                split='val',
                download=False,
                transform=transforms.Compose(                               #   组合多个图像变换
                    [
                        transforms.Resize(crop_size),                       #   通过Resize(size, interpolation=Image.BILINEAR)函数将输入的图像转换为期望的尺寸，此处期望的输出size=img_size
                        transforms.CenterCrop(crop_size),                    #   基于给定输入图像的中心，按照期望的尺寸（img_size）裁剪图像！！！解决了ImageNet数据集中个别样本size不符合3x256x256
                        transforms.ToTensor(),                                  #   将PIL图像或者ndArry数据转换为tensor                    
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]     
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'imagenetmixed10':
            # in_path = "/home/data/ImageNet"             #   ImageNet解压后的train和val所在的目录
            # in_info_path = "/home/data/ImageNet/info"
            # in_path = "/root/autodl-tmp/maggie/data/ImageNet"             #   ImageNet解压后的train和val所在的目录
            # in_info_path = "/root/autodl-tmp/maggie/data/ImageNet/info"
            in_path = "/home/huan1932/data/ImageNet"             #   ImageNet解压后的train和val所在的目录
            in_info_path = "/home/huan1932/data/ImageNet/info"

            in_hier = ImageNetHierarchy(in_path, in_info_path)                  

            superclass_wnid = common_superclass_wnid('mixed_10')            # group name: mixed_10
            class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

            # num_workers =4
            # batch_size =1

            custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
            # print("custom_dataset.__dict__.keys()",custom_dataset.__dict__.keys())
            # custom_dataset.__dict__.keys() dict_keys(['ds_name', 'data_path', 'num_classes', 'mean', 'std', 'transform_train', 'transform_test', 'custom_class', 'label_mapping', 'custom_class_args'])
            # train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, batch_size=batch_size)
            test_dataset = custom_dataset

            # if self._args.train_mode != 'cla-train':
            #     custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
            #     # print("custom_dataset.__dict__.keys()",custom_dataset.__dict__.keys())
            #     # custom_dataset.__dict__.keys() dict_keys(['ds_name', 'data_path', 'num_classes', 'mean', 'std', 'transform_train', 'transform_test', 'custom_class', 'label_mapping', 'custom_class_args'])
            #     # train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, batch_size=batch_size)

            # elif self._args.train_mode == 'cla-train':
            #     Maggie_TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
            #             transforms.Resize((256, 256)),
            #             transforms.CenterCrop(224),
            #             transforms.ToTensor()
            #         ])
            #     Maggie_TEST_TRANSFORMS_IMAGENET = transforms.Compose([
            #             transforms.Resize((256, 256)),
            #             transforms.CenterCrop(224),
            #             transforms.ToTensor()
            #         ])                    

            #     custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges, transform_train = Maggie_TRAIN_TRANSFORMS_IMAGENET, transform_test= Maggie_TEST_TRANSFORMS_IMAGENET)

            # test_dataset = custom_dataset

            return test_dataset

        elif self._args.dataset == 'lsun':
            os.makedirs("/home/data/maggie/lsun/20210413", exist_ok=True)
            test_dataset = MaggieLSUN(                                             
                "/home/data/maggie/lsun/20210413",
                #classes='train',   #   部分class由于无法解压缩包，不能训练
                #classes=['church_outdoor_train','classroom_train','tower_train'],   #   部分class由于无法解压缩包，不能训练
                classes=['church_outdoor_test','classroom_test','tower_test'],
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(self._args.img_size),                       
                        transforms.CenterCrop(self._args.img_size),                    
                        transforms.ToTensor(),                                  
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]     
                ),
                target_transform = None
            )
            return test_dataset    

        elif self._args.dataset == 'stl10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size                
            os.makedirs("/home/data/maggie/stl10", exist_ok=True)
            test_dataset = MaggieSTL10(                                             
                "/home/data/maggie/stl10",
                split='test',                                             #   从training.pt创建数据集
                download=False,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )  
            return test_dataset    
        
        elif self._args.dataset == 'svhn':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/maggie/svhn", exist_ok=True)
            test_dataset = MaggieSVHN(                                             
                "/home/data/maggie/svhn",
                split='test',                                             #   从training.pt创建数据集
                download=False,                                          #   自动从网上下载数据集
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                        # transforms.Resize(self._args.img_size), 
                        # transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )  
            return test_dataset  

class Array2Dataset:
    def __init__(self, args, x_ndarray, y_ndarray, ori_dataset: MaggieCIFAR10):
        print(f'将{args.dataset}的对抗样本数组x_ndarray,y_ndarray变换为Dataset')
        self._args = args
        self._x_ndarray = x_ndarray
        self._y_ndarray = y_ndarray
        self._ori_dataset_4_rep = copy.deepcopy(ori_dataset)
        self._ori_dataset_4_aug = copy.deepcopy(ori_dataset)

    def repdataset(self)->"MaggieDataset(torchvision.datasets.__dict__)":   # 不能和aug同时使用
        self._rep_dataset = self.__getrepdataset__()
        return self._rep_dataset

    def augdataset(self) ->"MaggieDataset(torchvision.datasets.__dict__)":
        self._aug_dataset = self.__getaugdataset__()
        return self._aug_dataset
    
    def __getrepdataset__(self):

        # print("before rep, self._ori_dataset_4_rep.data:",self._ori_dataset_4_rep.data[:2])
        self._ori_dataset_4_rep.replacedataset(self._x_ndarray,self._y_ndarray)
        # print("after rep, self._ori_dataset_4_rep.data:",self._ori_dataset_4_rep.data[:2])
        # return rep_dataset
        return self._ori_dataset_4_rep
    
    def __getaugdataset__(self)->"MaggieDataset(torchvision.datasets.__dict__)":
        self._ori_dataset_4_aug.augmentdataset(self._x_ndarray,self._y_ndarray)
        # return aug_dataset
        return self._ori_dataset_4_aug

